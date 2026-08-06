"""Training pipeline."""

import gc
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .data import (
    PerResidueDataset,
    _seq_hash,
    build_embedding_cache,
    collate_fn,
    parse_fasta,
)
from .model import ETA
from .utils import authenticate_hf, load_esm3

DEFAULT_HPARAMS: Dict[str, Any] = {
    'proj_dim': 256,
    'num_layers': 4,
    'num_heads': 8,
    'ffn_dim': 512,
    'dropout': 0.1,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'embed_batch_size': 16,
    'max_seq_len': 1536,
    'max_epochs': 30,
    'patience': 10,
    'warmup_epochs': 3,
    'test_size': 0.20,
    'val_size': 0.20,
    'seed': 42,
}


def run_training(
    pos_fasta: str,
    neg_fasta: str,
    output_dir: str,
    hparams: Optional[Dict[str, Any]] = None,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Path, Dict[str, float]]:
    """
    Full training pipeline.

    Returns
    -------
    (checkpoint_path, test_metrics)
    """
    hp = {**DEFAULT_HPARAMS, **(hparams or {})}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed = hp['seed']

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    use_amp = device.type == 'cuda'
    print(f'Device: {device}', flush=True)

    # Lower embed batch size for non-CUDA to avoid OOM (OOM recovery will halve further if needed)
    if 'embed_batch_size' not in (hparams or {}) and device.type != 'cuda':
        hp['embed_batch_size'] = 4

    # ── Parse sequences ────────────────────────────────────────────────────────
    pos_records = parse_fasta(pos_fasta)
    neg_records = parse_fasta(neg_fasta)
    if not pos_records:
        raise ValueError(f'No valid sequences found in {pos_fasta}')
    if not neg_records:
        raise ValueError(f'No valid sequences found in {neg_fasta}')

    all_records = pos_records + neg_records
    sequences = [r[2] for r in all_records]
    labels    = [1] * len(pos_records) + [0] * len(neg_records)
    headers   = [r[0] for r in all_records]
    genes     = [r[1] for r in all_records]
    seq_ids   = list(range(len(sequences)))

    print(f'Positive: {len(pos_records)}  Negative: {len(neg_records)}  '
          f'Total: {len(all_records)}  Balance: {sum(labels)/len(labels):.3f}+',
          flush=True)

    # ── Stratified split ───────────────────────────────────────────────────────
    y = np.array(labels)
    idx = np.arange(len(y))
    idx_tv, idx_test = train_test_split(
        idx, test_size=hp['test_size'], stratify=y, random_state=seed)
    idx_train, idx_val = train_test_split(
        idx_tv, test_size=hp['val_size'], stratify=y[idx_tv], random_state=seed)
    print(f'Split — train: {len(idx_train)}  val: {len(idx_val)}  test: {len(idx_test)}',
          flush=True)

    # ── ESM3 embedding cache ───────────────────────────────────────────────────
    authenticate_hf(hf_token)

    seq_hash = _seq_hash(sequences)
    if cache_dir:
        cache_h5 = Path(cache_dir) / f'embeddings_{seq_hash}.h5'
    else:
        cache_h5 = out / f'.embedding_cache_{seq_hash}.h5'

    import h5py
    def _n_cached(p):
        try:
            with h5py.File(p, 'r') as f:
                return len(f.keys())
        except Exception:
            return -1

    if cache_h5.exists() and _n_cached(cache_h5) >= len(sequences):
        print(f'Using cached embeddings: {cache_h5}', flush=True)
    else:
        if cache_h5.exists():
            print(f'Cache incomplete ({_n_cached(cache_h5)}/{len(sequences)}) '
                  f'— resuming embedding', flush=True)
        esm3, tok, pad_id, model_dtype = load_esm3(device)
        build_embedding_cache(
            sequences, seq_ids, esm3, tok, pad_id, model_dtype, device,
            cache_h5, max_len=hp['max_seq_len'], batch_size=hp['embed_batch_size'],
        )
        del esm3
        gc.collect()
        if use_amp:
            torch.cuda.empty_cache()

    # ── DataLoaders ────────────────────────────────────────────────────────────
    def _loader(indices, shuffle):
        ds = PerResidueDataset(
            cache_h5,
            [seq_ids[i] for i in indices],
            [labels[i] for i in indices],
            hp['max_seq_len'],
        )
        return torch.utils.data.DataLoader(
            ds, batch_size=hp['batch_size'], shuffle=shuffle,
            collate_fn=collate_fn, num_workers=0, pin_memory=use_amp,
        )

    tr_dl = _loader(idx_train, shuffle=True)
    va_dl = _loader(idx_val,   shuffle=False)
    te_dl = _loader(idx_test,  shuffle=False)

    # ── Model ──────────────────────────────────────────────────────────────────
    model = ETA(
        proj_dim=hp['proj_dim'], num_layers=hp['num_layers'],
        num_heads=hp['num_heads'], ffn_dim=hp['ffn_dim'],
        dropout=hp['dropout'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'ETA model: {n_params:,} parameters', flush=True)

    opt = torch.optim.AdamW(
        model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.01, end_factor=1.0,
                total_iters=hp['warmup_epochs']),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=hp['max_epochs'] - hp['warmup_epochs'], eta_min=1e-6),
        ],
        milestones=[hp['warmup_epochs']],
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    best_val_auc, best_state, wait = 0.0, None, 0
    history = []
    t_total = time.time()

    # The best checkpoint is written to disk every time val-AUC improves, so an
    # interrupted session (e.g. a Colab timeout) still leaves the best-so-far model
    # on disk — no need to retrain from scratch.
    ckpt_path = out / 'best_model.pth'
    _meta = {'headers': headers, 'labels': labels, 'genes': genes,
             'idx_train': idx_train.tolist(), 'idx_val': idx_val.tolist(),
             'idx_test': idx_test.tolist(), 'n_params': n_params}

    print(f'\n{"Epoch":>5}  {"Loss":>8}  {"Val-AUC":>8}  {"LR":>9}  {"Time":>6}',
          flush=True)

    for epoch in range(hp['max_epochs']):
        t0 = time.time()
        model.train()
        run_loss, n_steps = 0.0, 0
        for xb, yb, mask in tr_dl:
            xb, yb, mask = xb.to(device), yb.to(device), mask.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device.type, enabled=use_amp):
                loss = crit(model(xb, mask), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item()
            n_steps += 1
        scheduler.step()

        model.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for xb, yb, mask in va_dl:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits = model(xb.to(device), mask.to(device))
                val_probs.append(torch.softmax(logits, 1)[:, 1].cpu())
                val_true.append(yb)
        val_auc = roc_auc_score(
            torch.cat(val_true).numpy(), torch.cat(val_probs).numpy())

        cur_lr = scheduler.get_last_lr()[0]
        epoch_loss = run_loss / n_steps
        history.append({'epoch': epoch + 1, 'loss': epoch_loss,
                         'val_auc': val_auc, 'lr': cur_lr})

        flag = ''
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
            flag = '  ✓'
            # persist best-so-far immediately (survives an interrupted run)
            torch.save({'state_dict': best_state, 'hparams': hp,
                        'val_auc': best_val_auc, 'metadata': _meta}, ckpt_path)
        else:
            wait += 1
            if wait >= hp['patience']:
                print(f'  Early stop at epoch {epoch + 1}', flush=True)
                break

        print(f'  {epoch + 1:3d}   {epoch_loss:.4f}   {val_auc:.4f}'
              f'   {cur_lr:.2e}   {time.time() - t0:.0f}s{flag}', flush=True)

    elapsed = (time.time() - t_total) / 60
    print(f'\nBest val AUC: {best_val_auc:.4f}  ({elapsed:.1f} min total)', flush=True)

    # ── Test evaluation ────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval()
    te_probs, te_preds, te_true = [], [], []
    with torch.no_grad():
        for xb, yb, mask in te_dl:
            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(xb.to(device), mask.to(device))
            te_probs.append(torch.softmax(logits, 1)[:, 1].cpu())
            te_preds.append(logits.argmax(1).cpu())
            te_true.append(yb)

    prob = torch.cat(te_probs).numpy()
    pred = torch.cat(te_preds).numpy()
    true = torch.cat(te_true).numpy()

    tn = int(((pred == 0) & (true == 0)).sum())
    tp = int(((pred == 1) & (true == 1)).sum())
    fn = int(((pred == 0) & (true == 1)).sum())
    fp = int(((pred == 1) & (true == 0)).sum())

    test_metrics = {
        'accuracy':        accuracy_score(true, pred),
        'roc_auc':         roc_auc_score(true, prob),
        'avg_precision':   average_precision_score(true, prob),
        'sensitivity':     tp / (tp + fn) if (tp + fn) > 0 else float('nan'),
        'specificity':     tn / (tn + fp) if (tn + fp) > 0 else float('nan'),
    }

    print(f'\n── Test Set Results ───────────────────────────────────────────')
    for k, v in test_metrics.items():
        print(f'  {k:20s}: {v:.4f}')

    # ── Save (final: same checkpoint, now with test metrics) ────────────────────
    torch.save({
        'state_dict':   best_state,
        'hparams':      hp,
        'test_metrics': test_metrics,
        'metadata':     _meta,
    }, ckpt_path)

    pd.DataFrame(history).to_csv(out / 'training_history.csv', index=False)
    pd.DataFrame([test_metrics]).to_csv(out / 'test_metrics.csv', index=False)

    print(f'\nCheckpoint        → {ckpt_path}')
    print(f'Training history  → {out / "training_history.csv"}')
    print(f'Test metrics      → {out / "test_metrics.csv"}')

    return ckpt_path, test_metrics
