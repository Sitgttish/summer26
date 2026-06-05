"""Five attention-weight analyses, generalized for any binary protein classification task."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

STANDARD_AA = list('ACDEFGHIKLMNPQRSTVWY')
VALID_AA = frozenset(STANDARD_AA)


def run_analyses(
    weights_list: List[np.ndarray],
    labels: List[Optional[int]],
    sequences: List[str],
    gene_names: List[str],
    output_dir: Path,
    k: int = 5,
    top_n_motifs: int = 20,
    n_bins: int = 50,
) -> None:
    """
    Run five attention-weight analyses and save PNG plots to output_dir.

    Analysis 1+2 : Amino acid attention + class enrichment
    Analysis 3   : High-attention k-mer motifs (top 15% per sequence)
    Analysis 4   : Gene-level attention heatmap (row-normalised)
    Analysis 5   : Positional attention profile (N→C)
    """
    output_dir = Path(output_dir)
    has_labels = any(l is not None for l in labels)
    print('Running attention analyses ...', flush=True)

    # ── 1 + 2: Per-amino-acid attention ───────────────────────────────────────
    aa_w: dict = {aa: {'pos': [], 'neg': [], 'all': []} for aa in STANDARD_AA}
    for weights, label, seq in zip(weights_list, labels, sequences):
        L = min(len(seq), len(weights))
        for pos_idx, aa in enumerate(seq[:L]):
            if aa not in VALID_AA:
                continue
            w = float(weights[pos_idx])
            aa_w[aa]['all'].append(w)
            if label == 1:
                aa_w[aa]['pos'].append(w)
            elif label == 0:
                aa_w[aa]['neg'].append(w)

    mean_all = {aa: np.mean(aa_w[aa]['all']) if aa_w[aa]['all'] else 0.0
                for aa in STANDARD_AA}
    mean_pos = {aa: np.mean(aa_w[aa]['pos']) if aa_w[aa]['pos'] else 0.0
                for aa in STANDARD_AA}
    mean_neg = {aa: np.mean(aa_w[aa]['neg']) if aa_w[aa]['neg'] else 0.0
                for aa in STANDARD_AA}
    log_ratio = {
        aa: np.log2((mean_pos[aa] + 1e-10) / (mean_neg[aa] + 1e-10))
        for aa in STANDARD_AA
    }
    sorted_aa = sorted(STANDARD_AA, key=lambda a: log_ratio[a], reverse=True)

    if has_labels:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        x, w = np.arange(len(STANDARD_AA)), 0.35
        axes[0].bar(x - w/2, [mean_pos[a] for a in STANDARD_AA], w,
                    label='Positive (+)', color='#DD8452')
        axes[0].bar(x + w/2, [mean_neg[a] for a in STANDARD_AA], w,
                    label='Negative (−)', color='#4C72B0')
        axes[0].set_xticks(x); axes[0].set_xticklabels(STANDARD_AA)
        axes[0].set_ylabel('Mean attention weight')
        axes[0].set_title('Analysis 1: Attention per amino acid (by class)')
        axes[0].legend(); axes[0].grid(axis='y', alpha=0.3)

        colors = ['#c0392b' if log_ratio[a] > 0 else '#2980b9' for a in sorted_aa]
        axes[1].bar(sorted_aa, [log_ratio[a] for a in sorted_aa], color=colors)
        axes[1].axhline(0, color='black', linewidth=0.8)
        axes[1].set_ylabel('log₂(pos / neg attention)')
        axes[1].set_title('Analysis 2: Positive-class attention enrichment per amino acid')
        axes[1].grid(axis='y', alpha=0.3)
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(STANDARD_AA))
        ax.bar(x, [mean_all[a] for a in STANDARD_AA], color='#55A868')
        ax.set_xticks(x); ax.set_xticklabels(STANDARD_AA)
        ax.set_ylabel('Mean attention weight')
        ax.set_title('Analysis 1: Attention per amino acid')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'attn_aa_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  → attn_aa_analysis.png')

    # ── 3: High-attention k-mer motifs ────────────────────────────────────────
    kmers: dict = defaultdict(list)
    for weights, label, seq in zip(weights_list, labels, sequences):
        L = min(len(seq), len(weights))
        if L < k:
            continue
        thresh = np.percentile(weights[:L], 85)
        bucket = str(label) if label is not None else 'all'
        for pos_idx in range(L - k + 1):
            if weights[pos_idx:pos_idx + k].mean() >= thresh:
                mer = seq[pos_idx:pos_idx + k]
                if all(aa in VALID_AA for aa in mer):
                    kmers[bucket].append(mer)

    buckets = sorted(kmers.keys())
    label_map = {'1': 'Positive', '0': 'Negative', 'all': 'All sequences'}
    color_map = {'1': '#DD8452', '0': '#4C72B0', 'all': '#55A868'}
    ncols = max(1, len(buckets))
    fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 6),
                              squeeze=False)
    for ax, bucket in zip(axes[0], buckets):
        top = Counter(kmers[bucket]).most_common(top_n_motifs)
        if top:
            names, counts = zip(*top)
            ax.barh(list(names)[::-1], list(counts)[::-1],
                    color=color_map.get(bucket, '#888'))
        ax.set_xlabel('Count')
        ax.set_title(f'Analysis 3: Top-{top_n_motifs} high-attention {k}-mers\n'
                     f'{label_map.get(bucket, bucket)}')
        ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attn_motifs.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  → attn_motifs.png')

    # ── 4: Gene-level attention heatmap ───────────────────────────────────────
    gene_aa_data: dict = defaultdict(lambda: defaultdict(list))
    for weights, gene, seq in zip(weights_list, gene_names, sequences):
        if not gene:
            continue
        L = min(len(seq), len(weights))
        for pos_idx, aa in enumerate(seq[:L]):
            if aa in VALID_AA:
                gene_aa_data[gene][aa].append(float(weights[pos_idx]))

    top_genes = sorted(
        gene_aa_data.keys(),
        key=lambda g: sum(len(v) for v in gene_aa_data[g].values()),
        reverse=True,
    )[:25]

    if top_genes:
        mat = pd.DataFrame(index=top_genes, columns=STANDARD_AA, dtype=float)
        for gene in top_genes:
            for aa in STANDARD_AA:
                vals = gene_aa_data[gene][aa]
                mat.loc[gene, aa] = np.mean(vals) if vals else 0.0
        mat_norm = mat.div(mat.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(14, max(6, len(top_genes) * 0.4)))
        sns.heatmap(mat_norm.astype(float), cmap='YlOrRd', ax=ax,
                    linewidths=0.3,
                    cbar_kws={'label': 'Relative attention share'})
        ax.set_title('Analysis 4: Gene-level attention per amino acid type (row-normalised)')
        ax.set_xlabel('Amino acid')
        ax.set_ylabel('Gene')
        plt.tight_layout()
        plt.savefig(output_dir / 'attn_gene_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('  → attn_gene_heatmap.png')

    # ── 5: Positional attention profile ───────────────────────────────────────
    bins = np.linspace(0, 1, n_bins + 1)
    bcenters = (bins[:-1] + bins[1:]) / 2
    density: dict  = defaultdict(lambda: np.zeros(n_bins))
    bin_counts: dict = defaultdict(lambda: np.zeros(n_bins))

    for weights, label in zip(weights_list, labels):
        L = len(weights)
        if L < 2:
            continue
        rel = np.arange(L) / (L - 1)
        idx_b = np.clip(np.digitize(rel, bins) - 1, 0, n_bins - 1)
        bucket = str(label) if label is not None else 'all'
        for b, ww in zip(idx_b, weights):
            density[bucket][b] += ww
            bin_counts[bucket][b] += 1

    means = {
        bucket: np.divide(density[bucket], bin_counts[bucket],
                           where=bin_counts[bucket] > 0)
        for bucket in density
    }

    fig, ax = plt.subplots(figsize=(12, 5))
    for bucket, mean_w in sorted(means.items()):
        ax.plot(bcenters, mean_w, linewidth=2,
                color=color_map.get(bucket, '#888'),
                label=label_map.get(bucket, bucket))
    ax.axvline(0.2, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0.8, color='black', linestyle='--', linewidth=0.8, alpha=0.5,
               label='N/C-term boundary (20%)')
    ax.set_xlabel('Relative position (0 = N-terminus, 1 = C-terminus)')
    ax.set_ylabel('Mean attention weight')
    ax.set_title('Analysis 5: Attention concentration by sequence position')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'attn_position.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('  → attn_position.png')
    print(f'Analyses saved to {output_dir}')
