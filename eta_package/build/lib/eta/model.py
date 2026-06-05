import math

import torch
import torch.nn as nn


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class ETA(nn.Module):
    """
    ESM3-based Transformer Attention binary classifier.

    Pipeline:
        ESM3 per-residue embeddings  (B, L, 1536)
        → Linear + LayerNorm + SinPosEnc       (B, L, proj_dim)
        → N × Pre-LN TransformerEncoderLayer   (B, L, proj_dim)
        → Learned attention pooling            (B, proj_dim)
        → LayerNorm + Dropout + Linear         (B, 2)
    """

    def __init__(
        self,
        input_dim: int = 1536,
        proj_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert proj_dim % num_heads == 0, 'proj_dim must be divisible by num_heads'

        self.proj = nn.Linear(input_dim, proj_dim)
        self.norm_in = nn.LayerNorm(proj_dim)
        self.pos_enc = SinusoidalPosEnc(proj_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.attn_q = nn.Parameter(torch.randn(proj_dim) * 0.02)
        self.norm_out = nn.LayerNorm(proj_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(proj_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_pooled: bool = False,
        return_weights: bool = False,
    ):
        h = self.norm_in(self.proj(x))
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=padding_mask)

        scores = (h @ self.attn_q) / (self.attn_q.shape[0] ** 0.5)
        if padding_mask is not None:
            scores = scores.masked_fill(padding_mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)          # (B, L)
        pooled = (attn.unsqueeze(-1) * h).sum(dim=1) # (B, proj_dim)

        if return_pooled:
            return pooled
        logits = self.fc(self.drop(self.norm_out(pooled)))
        if return_weights:
            return logits, attn
        return logits
