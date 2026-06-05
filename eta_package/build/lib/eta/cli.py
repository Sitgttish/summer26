"""Command-line interface for ETA."""

import argparse
import os
import sys


def _resolve_token(args) -> str | None:
    return getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN')


def main():
    parser = argparse.ArgumentParser(
        prog='eta',
        description=(
            'ETA — ESM3-based Transformer Attention classifier\n'
            'for protein binary classification.\n\n'
            'Examples:\n'
            '  eta --train positive.fasta negative.fasta ./model_output/\n'
            '  eta --eval  model_output/best_model.pth new_seqs.fasta ./results.csv\n'
            '  eta --eval  best_model.pth seqs.fasta ./results.csv --gene-analyze'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        '--train',
        nargs=3,
        metavar=('POS_FASTA', 'NEG_FASTA', 'OUTPUT_DIR'),
        help='Train a new ETA model.',
    )
    mode.add_argument(
        '--eval',
        nargs=3,
        metavar=('MODEL_PATH', 'SEQUENCES_FASTA', 'OUTPUT_CSV'),
        help='Run inference with a saved checkpoint.',
    )

    # ── Shared ────────────────────────────────────────────────────────────────
    parser.add_argument(
        '--hf-token', metavar='TOKEN',
        help='HuggingFace token for ESM3 (or set HF_TOKEN env var).',
    )
    parser.add_argument(
        '--cache-dir', metavar='DIR',
        help='Directory to store/reuse ESM3 embedding cache (training only).',
    )
    parser.add_argument(
        '--embed-batch-size', type=int, default=None, metavar='N',
        help='ESM3 embedding batch size (default: 16 train / 8 eval). '
             'Reduce if you hit OOM during the embedding step.',
    )
    parser.add_argument(
        '--max-seq-len', type=int, default=1536, metavar='N',
        help='Truncate sequences to this length (default: 1536).',
    )

    # ── Training hyperparameters ──────────────────────────────────────────────
    tr = parser.add_argument_group('Training hyperparameters')
    tr.add_argument('--epochs',       type=int,   default=30,   metavar='N')
    tr.add_argument('--patience',     type=int,   default=10,   metavar='N')
    tr.add_argument('--batch-size',   type=int,   default=64,   metavar='N',
                    help='ETA training/inference batch size (default: 64).')
    tr.add_argument('--lr',           type=float, default=3e-4, metavar='FLOAT')
    tr.add_argument('--weight-decay', type=float, default=1e-4, metavar='FLOAT')
    tr.add_argument('--proj-dim',     type=int,   default=256,  metavar='N')
    tr.add_argument('--num-layers',   type=int,   default=4,    metavar='N')
    tr.add_argument('--num-heads',    type=int,   default=8,    metavar='N')
    tr.add_argument('--ffn-dim',      type=int,   default=512,  metavar='N')
    tr.add_argument('--dropout',      type=float, default=0.1,  metavar='FLOAT')
    tr.add_argument('--seed',         type=int,   default=42,   metavar='N')

    # ── Eval / analysis options ────────────────────────────────────────────────
    ev = parser.add_argument_group('Evaluation options')
    ev.add_argument(
        '--gene-analyze', action='store_true',
        help='Run five attention-weight analyses after inference and save plots.',
    )
    ev.add_argument(
        '--analyze-dir', metavar='DIR',
        help='Directory for analysis outputs (default: <OUTPUT_CSV parent>/analysis/).',
    )

    args = parser.parse_args()
    hf_token = _resolve_token(args)

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.train:
        pos_fasta, neg_fasta, output_dir = args.train
        hparams = {
            'proj_dim':         args.proj_dim,
            'num_layers':       args.num_layers,
            'num_heads':        args.num_heads,
            'ffn_dim':          args.ffn_dim,
            'dropout':          args.dropout,
            'lr':               args.lr,
            'weight_decay':     args.weight_decay,
            'batch_size':       args.batch_size,
            'embed_batch_size': args.embed_batch_size or 16,
            'max_seq_len':      args.max_seq_len,
            'max_epochs':       args.epochs,
            'patience':         args.patience,
            'seed':             args.seed,
        }
        from .train import run_training
        run_training(
            pos_fasta=pos_fasta,
            neg_fasta=neg_fasta,
            output_dir=output_dir,
            hparams=hparams,
            hf_token=hf_token,
            cache_dir=args.cache_dir,
        )

    # ── Eval ──────────────────────────────────────────────────────────────────
    elif args.eval:
        model_path, sequences_fasta, output_csv = args.eval
        from .eval import run_eval
        run_eval(
            model_path=model_path,
            sequences_fasta=sequences_fasta,
            output_path=output_csv,
            hf_token=hf_token,
            gene_analyze=args.gene_analyze,
            analyze_dir=args.analyze_dir,
            embed_batch_size=args.embed_batch_size or 8,
        )


if __name__ == '__main__':
    main()
