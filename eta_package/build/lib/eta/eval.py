"""Inference pipeline."""

import gc
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

from .data import _seq_hash, build_embedding_cache, parse_fasta
from .model import ETA
from .utils import authenticate_hf, load_esm3


def run_eval(
    model_path: str,
    sequences_fasta: str,
    output_path: str,
    hf_token: Optional[str] = None,
    gene_analyze: bool = False,
    analyze_dir: Optional[str] = None,
    embed_batch_size: int = 8,
) -> pd.DataFrame:
    """
    Run inference on sequences_fasta using a saved ETA checkpoint.

    Saves a CSV to output_path with columns:
      header, gene, prob_positive, predicted_label, [true_label, correct]

    If --gene-analyze: also saves attention analysis plots to analyze_dir.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device}', flush=True)

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    hp   = ckpt['hparams']
    model = ETA(
        proj_dim=hp['proj_dim'], num_layers=hp['num_layers'],
        num_heads=hp['num_heads'], ffn_dim=hp['ffn_dim'],
        dropout=hp['dropout'],
    ).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f'Model loaded from {model_path}', flush=True)

    # ── Parse input sequences ──────────────────────────────────────────────────
    records = parse_fasta(sequences_fasta)
    if not records:
        raise ValueError(f'No valid sequences found in {sequences_fasta}')

    headers    = [r[0] for r in records]
    genes      = [r[1] for r in records]
    sequences  = [r[2] for r in records]
    raw_labels = [r[3] for r in records]
    has_labels = all(l is not None for l in raw_labels)
    seq_ids    = list(range(len(sequences)))

    print(f'Input: {len(sequences)} sequences  (labels: {"yes" if has_labels else "no"})',
          flush=True)

    # ── ESM3 embedding ─────────────────────────────────────────────────────────
    authenticate_hf(hf_token)
    esm3, tok, pad_id, model_dtype = load_esm3(device)

    out_parent = Path(output_path).parent
    out_parent.mkdir(parents=True, exist_ok=True)
    tmp_h5 = out_parent / f'.eta_eval_{_seq_hash(sequences)}.h5'

    if not tmp_h5.exists():
        build_embedding_cache(
            sequences, seq_ids, esm3, tok, pad_id, model_dtype, device,
            tmp_h5, max_len=hp.get('max_seq_len', 1536),
            batch_size=embed_batch_size,
        )
    else:
        print(f'Using cached embeddings: {tmp_h5}', flush=True)

    del esm3
    gc.collect()
    if use_amp:
        torch.cuda.empty_cache()

    # ── Inference ──────────────────────────────────────────────────────────────
    max_len = hp.get('max_seq_len', 1536)
    all_probs, all_preds, all_weights = [], [], []
    t0 = time.time()
    print(f'Running inference on {len(sequences)} sequences ...', flush=True)

    with h5py.File(tmp_h5, 'r') as hf, torch.no_grad():
        for start in range(0, len(seq_ids), 64):
            batch_ids = seq_ids[start:start + 64]
            embs = [torch.from_numpy(hf[str(k)][:max_len].astype('float32'))
                    for k in batch_ids]
            lengths = [e.shape[0] for e in embs]
            max_L = max(lengths)
            D = embs[0].shape[1]
            padded = torch.zeros(len(embs), max_L, D)
            mask   = torch.ones(len(embs), max_L, dtype=torch.bool)
            for i, (e, L) in enumerate(zip(embs, lengths)):
                padded[i, :L] = e
                mask[i, :L]   = False

            with torch.amp.autocast(device.type, enabled=use_amp):
                if gene_analyze:
                    logits, w = model(padded.to(device), mask.to(device),
                                      return_weights=True)
                    w_np = w.cpu().numpy()
                    m_np = mask.numpy()
                    for i in range(len(batch_ids)):
                        real_L = int((~m_np[i]).sum())
                        all_weights.append(w_np[i, :real_L])
                else:
                    logits = model(padded.to(device), mask.to(device))

            probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
            all_probs.extend(probs.tolist())
            all_preds.extend((probs >= 0.5).astype(int).tolist())

    print(f'Inference done ({time.time() - t0:.1f}s)', flush=True)

    # ── Results dataframe ──────────────────────────────────────────────────────
    results = pd.DataFrame({
        'header':          headers,
        'gene':            genes,
        'prob_positive':   all_probs,
        'predicted_label': all_preds,
    })
    if has_labels:
        results['true_label'] = raw_labels
        results['correct']    = (results['predicted_label'] == results['true_label']).astype(int)

    # ── Report metrics ─────────────────────────────────────────────────────────
    n_pos = results['predicted_label'].sum()
    print(f'\n── Inference Results ─────────────────────────────────────────')
    print(f'  Total sequences    : {len(results)}')
    print(f'  Predicted positive : {n_pos} ({n_pos / len(results):.3f})')
    print(f'  Mean probability   : {results["prob_positive"].mean():.4f}')

    if has_labels:
        true = np.array(raw_labels, dtype=int)
        prob = np.array(all_probs)
        pred = np.array(all_preds)
        cm = confusion_matrix(true, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics = {
            'accuracy':      accuracy_score(true, pred),
            'roc_auc':       roc_auc_score(true, prob),
            'avg_precision': average_precision_score(true, prob),
            'sensitivity':   tp / (tp + fn) if (tp + fn) > 0 else float('nan'),
            'specificity':   tn / (tn + fp) if (tn + fp) > 0 else float('nan'),
        }
        print(f'\n── Metrics ───────────────────────────────────────────────────')
        for k, v in metrics.items():
            print(f'  {k:20s}: {v:.4f}')
        pd.DataFrame([metrics]).to_csv(
            Path(output_path).parent / 'eval_metrics.csv', index=False)
        print(f'Metrics saved → {Path(output_path).parent / "eval_metrics.csv"}')
    else:
        print(f'\n  (No labels in FASTA — add |label=1 or |label=0 to headers for metrics)')

    # ── Save predictions ───────────────────────────────────────────────────────
    results.to_csv(output_path, index=False)
    print(f'Predictions saved → {output_path}')

    # ── Attention analyses ─────────────────────────────────────────────────────
    if gene_analyze:
        if not all_weights:
            print('Warning: no attention weights collected — skipping analysis.')
        else:
            from .analysis import run_analyses
            adir = Path(analyze_dir) if analyze_dir \
                   else Path(output_path).parent / 'analysis'
            adir.mkdir(parents=True, exist_ok=True)
            run_analyses(
                weights_list=all_weights,
                labels=raw_labels if has_labels else [None] * len(sequences),
                sequences=sequences,
                gene_names=genes,
                output_dir=adir,
            )

    return results
