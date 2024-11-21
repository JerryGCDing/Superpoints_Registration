# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------

import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [n_cls_token+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if n_cls_token > 0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# MAE: https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


class NerfPositionalEncoding(torch.nn.Module):
    def __init__(self, depth=10, sine_type='lin_sine'):
        '''
        out_dim = in_dim * depth * 2
        '''
        super().__init__()
        if sine_type == 'lin_sine':
            self.bases = [i + 1 for i in range(depth)]
        elif sine_type == 'exp_sine':
            self.bases = [2 ** i for i in range(depth)]
        # print(f'using {sine_type} as positional encoding for queries.')

    @torch.no_grad()
    def forward(self, inputs):
        out = torch.cat(
            [torch.sin(i * math.pi * inputs) for i in self.bases] + [torch.cos(i * math.pi * inputs) for i in
                                                                     self.bases], axis=-1)
        assert torch.isnan(out).any() == False
        return out


# ----------------------------------------------------------
# RoPE2D: RoPE implementation in 2D
# ----------------------------------------------------------
try:
    from mapo.nn.embedder.curope import cuRoPE2D

    RoPE2D = cuRoPE2D
    print('cuda-compiled RoPE2d successfully loaded!!')
except ImportError:
    print('Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead')


    class RoPE2D(torch.nn.Module):

        def __init__(self, freq=100.0, F0=1.0):
            super().__init__()
            self.base = freq
            self.F0 = F0
            self.cache = {}

        def get_cos_sin(self, D, seq_len, device, dtype):
            if (D, seq_len, device, dtype) not in self.cache:
                inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
                t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
                freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
                freqs = torch.cat((freqs, freqs), dim=-1)
                cos = freqs.cos()  # (Seq, Dim)
                sin = freqs.sin()
                self.cache[D, seq_len, device, dtype] = (cos, sin)
            return self.cache[D, seq_len, device, dtype]

        @staticmethod
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rope1d(self, tokens, pos1d, cos, sin):
            assert pos1d.ndim == 2
            cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
            sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
            return (tokens * cos) + (self.rotate_half(tokens) * sin)

        def forward(self, tokens, positions):
            """
            input:
                * tokens: batch_size x nheads x ntokens x dim
                * positions: batch_size x ntokens x 2 (y and x position of each token)
            output:
                * tokens after appplying RoPE2D (batch_size x nheads x ntokens x dim)
            """
            assert tokens.size(3) % 2 == 0, "number of dimensions should be a multiple of two"
            D = tokens.size(3) // 2
            assert positions.ndim == 3 and positions.shape[-1] == 2  # Batch, Seq, 2
            cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
            # split features into two along the feature dimension, and apply rope1d on each half
            y, x = tokens.chunk(2, dim=-1)
            y = self.apply_rope1d(y, positions[:, :, 0], cos, sin)
            x = self.apply_rope1d(x, positions[:, :, 1], cos, sin)
            tokens = torch.cat((y, x), dim=-1)
            return tokens


class RoPE3D(torch.nn.Module):

    def __init__(self, freq=100.0, F0=1.0):
        super().__init__()
        self.base = freq
        self.F0 = F0
        self.cache = {}

    def get_cos_sin(self, D, seq_len, device, dtype):
        if (D, seq_len, device, dtype) not in self.cache:
            inv_freq = 1.0 / (self.base ** (torch.arange(0, D, 2).float().to(device) / D))
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, inv_freq).to(dtype)
            freqs = torch.cat((freqs, freqs), dim=-1)
            cos = freqs.cos()  # (Seq, Dim)
            sin = freqs.sin()
            self.cache[D, seq_len, device, dtype] = (cos, sin)
        return self.cache[D, seq_len, device, dtype]

    @staticmethod
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rope1d(self, tokens, pos1d, cos, sin):
        assert pos1d.ndim == 2
        cos = torch.nn.functional.embedding(pos1d, cos)[:, None, :, :]
        sin = torch.nn.functional.embedding(pos1d, sin)[:, None, :, :]
        return (tokens * cos) + (self.rotate_half(tokens) * sin)

    def forward(self, tokens, positions):
        """
        input:
            * tokens: batch_size x nheads x ntokens x dim
            * positions: batch_size x ntokens x 3 (x, y and z position of each token)
        output:
            * tokens after appplying RoPE3D (batch_size x nheads x ntokens x dim)
        """
        assert tokens.size(3) % 3 == 0, "number of dimensions should be a multiple of three"
        D = tokens.size(3) // 3

        assert positions.ndim == 3 and positions.shape[-1] == 3  # Batch, Seq, 3
        cos, sin = self.get_cos_sin(D, int(positions.max()) + 1, tokens.device, tokens.dtype)
        # split features into three along the feature dimension, and apply rope1d on each
        x, y, z = tokens.chunk(3, dim=-1)
        x = self.apply_rope1d(x, positions[:, :, 0], cos, sin)
        y = self.apply_rope1d(y, positions[:, :, 1], cos, sin)
        z = self.apply_rope1d(z, positions[:, :, 2], cos, sin)
        tokens = torch.cat((x, y, z), dim=-1)
        return tokens


class PositionEmbeddingCoordsSine(nn.Module):
    """Similar to transformer's position encoding, but generalizes it to
    arbitrary dimensions and continuous coordinates.

    Args:
        n_dim: Number of input dimensions, e.g. 2 for image coordinates.
        d_model: Number of dimensions to encode into
        temperature:
        scale:
    """

    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=None):
        super().__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None:
            scale = 1.0
        self.scale = scale * 2 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: Point positions (*, d_in)

        Returns:
            pos_emb (*, d_out)
        """
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        # Pad unused dimensions with zeros
        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb
