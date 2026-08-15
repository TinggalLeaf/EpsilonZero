"""Transformer policy-value network for Gomoku (CPU friendly).

Input:  (B, 4, N, N) board planes  -> tokens of size N*N
Output: policy logits (B, N*N), value (B, 1) in [-1, 1]
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoardEmbedding(nn.Module):
    def __init__(self, board_size: int, d_model: int, in_planes: int = 4):
        super().__init__()
        self.board_size = board_size
        self.proj = nn.Linear(in_planes, d_model)
        # learnable 2D positional embedding, interpolated when board size changes
        self.pos = nn.Parameter(torch.zeros(1, board_size * board_size, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def positional(self, n: int) -> torch.Tensor:
        """Return positional embedding for an n x n board (bilinear interp if needed)."""
        base = self.board_size
        if n == base:
            return self.pos
        pe = self.pos.reshape(1, base, base, -1).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(n, n), mode="bilinear", align_corners=False)
        return pe.permute(0, 2, 3, 1).reshape(1, n * n, -1)

    def forward(self, x: torch.Tensor, n: int) -> torch.Tensor:
        # x: (B, 4, n, n)
        b = x.shape[0]
        x = x.reshape(b, 4, n * n).transpose(1, 2)  # (B, n*n, 4)
        return self.proj(x) + self.positional(n)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(a)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class GomokuTransformer(nn.Module):
    def __init__(self, board_size: int = 15, d_model: int = 128, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 256, dropout: float = 0.1):
        super().__init__()
        self.board_size = board_size
        self.embed = BoardEmbedding(board_size, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        # policy head: per-token logit
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        # value head: pooled
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1), nn.Tanh())

    def forward(self, x: torch.Tensor):
        # x: (B, 4, n, n); n may differ from self.board_size
        n = x.shape[-1]
        h = self.embed(x, n)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        policy = self.policy_head(h).squeeze(-1)          # (B, n*n)
        value = self.value_head(h.mean(dim=1))            # (B, 1)
        return policy, value

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax over legal moves only. mask: 1 = legal."""
    logits = logits.masked_fill(mask <= 0, -1e9)
    return F.softmax(logits, dim=-1)
