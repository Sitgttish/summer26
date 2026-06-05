# ETAP — ESM3-based Transformer Attention Protein classifier

Binary protein sequence classifier built on ESM3 per-residue embeddings with a learned attention pooling layer.  
Designed for any study requiring positive/negative classification of protein sequences.

## Installation

```bash
pip install etap-clf
```

Or from source:
```bash
pip install "git+https://github.com/Sitgttish/summer26.git#subdirectory=eta_package"
```

ESM3 is a gated model. Before first use:
1. Accept the license at https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
2. Get a token at https://huggingface.co/settings/tokens
3. Pass it via `--hf-token` or set `HF_TOKEN` in your environment.

## Usage

### Training

```bash
etap --train positive.fasta negative.fasta ./model_output/
```

Outputs saved to `./model_output/`:
- `best_model.pth` — model checkpoint
- `training_history.csv` — epoch-level loss and val-AUC
- `test_metrics.csv` — final accuracy, AUC, avg-precision, sensitivity, specificity

Optional flags:
```
--epochs 30          Max training epochs (default: 30)
--patience 10        Early stopping patience on val-AUC (default: 10)
--batch-size 64      ETA training batch size (default: 64)
--embed-batch-size 16  ESM3 embedding batch size; reduce if OOM (default: 16)
--lr 3e-4            Learning rate (default: 3e-4)
--proj-dim 256       Hidden dimension (default: 256)
--num-layers 4       Transformer encoder layers (default: 4)
--cache-dir ./cache  Reuse ESM3 embedding cache across runs
--hf-token TOKEN     HuggingFace token
```

### Inference

```bash
etap --eval model_output/best_model.pth new_sequences.fasta ./results.csv
```

Output CSV columns: `header, gene, prob_positive, predicted_label`  
If FASTA headers contain `|label=1` or `|label=0`, full metrics are reported automatically.

### Attention analysis (optional)

```bash
etap --eval best_model.pth sequences.fasta ./results.csv --gene-analyze
```

Saves five plots to `./analysis/`:
1. `attn_aa_analysis.png` — mean attention per amino acid type (+ class enrichment if labels present)
2. `attn_motifs.png` — top high-attention 5-mer motifs
3. `attn_gene_heatmap.png` — gene-level attention heatmap
4. `attn_position.png` — positional attention profile (N→C terminus)

## FASTA header conventions

**Gene name** is extracted automatically:
- UniProt format `>sp|P12345|GENE_HUMAN` → gene = `GENE`
- Generic: first token before space/`|`/`_`

**Labels** (for metric reporting in eval):
```
>SEQID|label=1    ferroptosis-positive
>SEQID|label=0    negative control
```

## Python API

```python
from etap import ETA, run_training, run_eval

# Training
ckpt_path, metrics = run_training(
    pos_fasta='positive.fasta',
    neg_fasta='negative.fasta',
    output_dir='./output/',
    hf_token='hf_...',
)

# Inference
results = run_eval(
    model_path='./output/best_model.pth',
    sequences_fasta='new_seqs.fasta',
    output_path='./results.csv',
    gene_analyze=True,
)
```
