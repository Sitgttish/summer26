"""Shared utilities: HuggingFace auth, ESM3 loading."""

import os
from typing import Optional

import torch


def authenticate_hf(hf_token: Optional[str] = None) -> None:
    """Authenticate with HuggingFace. Raises RuntimeError with helpful message on failure."""
    from huggingface_hub import login, whoami

    token = hf_token or os.environ.get('HF_TOKEN')
    if token:
        login(token=token, add_to_git_credential=False)
        return
    try:
        whoami()
    except Exception:
        raise RuntimeError(
            '\nHuggingFace authentication required to access ESM3.\n'
            'Steps:\n'
            '  1. Accept the model license at:\n'
            '     https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1\n'
            '  2. Get a token at: https://huggingface.co/settings/tokens\n'
            '  3. Pass it via --hf-token TOKEN or set the HF_TOKEN environment variable.'
        )


def load_esm3(device: torch.device):
    """Load ESM3-small and return (model, tokenizer, pad_id, model_dtype)."""
    from esm.models.esm3 import ESM3
    from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

    print('Loading ESM3 (esm3_sm_open_v1) ...', flush=True)
    model = ESM3.from_pretrained('esm3_sm_open_v1').eval()
    # bfloat16 on all devices: halves model weight memory (~2.8 GB vs ~5.6 GB)
    # and prevents per-sequence attention OOM on long sequences
    model = model.to(torch.bfloat16).to(device)
    dtype = next(model.parameters()).dtype
    print(f'ESM3 ready  (dtype={dtype}, device={device})', flush=True)
    tok = EsmSequenceTokenizer()
    return model, tok, tok.pad_token_id, dtype
