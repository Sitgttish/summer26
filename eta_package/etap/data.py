"""FASTA parsing, ESM3 embedding cache, and PyTorch dataset."""

import gc
import hashlib
import re
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from Bio import SeqIO
from tqdm import tqdm

VALID_AA = frozenset('ACDEFGHIKLMNPQRSTVWY')
ESM3_DIM = 1536


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def _extract_gene(header: str, seq_id: str) -> str:
    """
    Best-effort gene name extraction from a FASTA header.

    Recognises:
      - UniProt/Swiss-Prot:  >sp|P12345|GENE_SPECIES ...  → GENE
      - UniParc / generic:   first token before space / | / _
    """
    # UniProt format: sp|ACC|GENE_SPECIES
    m = re.match(r'(?:sp|tr)\|[^|]+\|([^_\s]+)', header)
    if m:
        return m.group(1)
    # Fall back to first field of the ID
    return seq_id.split('|')[0].split('_')[0]


def parse_fasta(
    path: str | Path,
) -> List[Tuple[str, str, str, Optional[int]]]:
    """
    Parse a FASTA file and return a list of (header, gene, sequence, label_or_None).

    Header label convention (optional):
      >SEQID|label=1   or   >SEQID|label=0
    """
    records = []
    for rec in SeqIO.parse(str(path), 'fasta'):
        seq = ''.join(c for c in str(rec.seq).upper() if c in VALID_AA)
        if len(seq) < 10:
            continue
        header = rec.description
        gene = _extract_gene(header, rec.id)
        label: Optional[int] = None
        if '|label=1' in header:
            label = 1
        elif '|label=0' in header:
            label = 0
        records.append((header, gene, seq, label))
    return records


# ── Embedding cache ───────────────────────────────────────────────────────────

def _seq_hash(sequences: List[str]) -> str:
    h = hashlib.md5()
    for s in sequences:
        h.update(s.encode())
    return h.hexdigest()[:16]


def build_embedding_cache(
    sequences: List[str],
    seq_ids: List[int],
    esm3_model,
    tokenizer,
    pad_id: int,
    model_dtype,
    device: torch.device,
    cache_path: Path,
    max_len: int = 1536,
    batch_size: int = 16,
) -> None:
    """Compute per-residue ESM3 embeddings for all sequences and write to HDF5."""
    from esm.sdk.api import ESMProtein

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume support: skip sequences already embedded so an interrupted run
    # (e.g. a Colab timeout) continues instead of restarting from scratch.
    mode, existing = 'w', set()
    if cache_path.exists():
        try:
            with h5py.File(cache_path, 'r') as _hf:
                existing = set(_hf.keys())
            mode = 'a'
        except Exception:
            mode = 'w'          # unreadable/corrupt → rebuild
    if existing:
        pairs = [(s, sid) for s, sid in zip(sequences, seq_ids)
                 if str(sid) not in existing]
        print(f'Resuming cache: {len(existing)} done, {len(pairs)} to embed',
              flush=True)
        sequences = [p[0] for p in pairs]
        seq_ids = [p[1] for p in pairs]
    n = len(sequences)
    if n == 0:
        print('All sequences already embedded.', flush=True)
        return
    print(f'Embedding {n} sequences → {cache_path}', flush=True)

    with h5py.File(cache_path, mode) as hf:
        bs = batch_size
        i = 0
        pbar = tqdm(total=n, desc='ESM3 embedding', unit='seq')
        while i < n:
            batch_seqs = sequences[i:i + bs]
            batch_ids = seq_ids[i:i + bs]
            try:
                token_list, actual_lens = [], []
                for seq in batch_seqs:
                    pt = esm3_model.encode(ESMProtein(sequence=seq[:max_len]))
                    token_list.append(pt.sequence)
                    actual_lens.append(min(len(seq), max_len))

                max_tok = max(t.shape[0] for t in token_list)
                padded = torch.stack([
                    F.pad(t, (0, max_tok - t.shape[0]), value=pad_id)
                    for t in token_list
                ]).to(device)

                with torch.inference_mode():
                    with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                        out = esm3_model(sequence_tokens=padded)

                for j, (sid, L) in enumerate(zip(batch_ids, actual_lens)):
                    key = str(sid)
                    if key in hf:
                        continue
                    emb = out.embeddings[j, 1:L + 1, :].float().cpu().numpy()
                    hf.create_dataset(
                        key, data=emb.astype('float16'),
                        compression='gzip', compression_opts=4,
                    )

                i += bs
                pbar.update(len(batch_seqs))

            except (torch.cuda.OutOfMemoryError, RuntimeError, MemoryError) as e:
                if isinstance(e, RuntimeError) and 'out of memory' not in str(e).lower():
                    raise
                if bs == 1:
                    raise RuntimeError(
                        'Out of memory even with a single sequence. '
                        'Try --max-seq-len 512 to truncate long sequences.'
                    ) from e
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif device.type == 'mps':
                    torch.mps.empty_cache()
                gc.collect()
                bs = max(1, bs // 2)
                tqdm.write(f'OOM — reducing embedding batch size to {bs}')
        pbar.close()

    size_gb = cache_path.stat().st_size / 1e9
    print(f'Cache saved ({size_gb:.2f} GB): {cache_path}', flush=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PerResidueDataset(torch.utils.data.Dataset):
    """Lazy HDF5 reader; one file handle per process, opened on first access."""

    def __init__(
        self,
        h5_path: str | Path,
        seq_ids: List[int],
        labels: List[int],
        max_len: int = 1536,
    ):
        self.h5_path = str(h5_path)
        self.seq_ids = seq_ids
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.max_len = max_len
        self._h5 = None

    def _file(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    def __len__(self):
        return len(self.seq_ids)

    def __getitem__(self, i):
        emb = torch.from_numpy(self._file()[str(self.seq_ids[i])][:].astype('float32'))
        if emb.shape[0] > self.max_len:
            emb = emb[:self.max_len]
        return emb, self.labels[i]


def collate_fn(batch):
    """Pad variable-length sequences; return (padded, labels, bool_mask)."""
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    max_L, D = max(lengths), seqs[0].shape[1]
    padded = torch.zeros(len(seqs), max_L, D)
    mask = torch.ones(len(seqs), max_L, dtype=torch.bool)  # True = pad
    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L] = s
        mask[i, :L] = False
    return padded, torch.stack(list(labels)), mask
