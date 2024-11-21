import collections.abc

import torch
import torch.nn as nn

from itertools import repeat
from typing import Optional, List
from torch.nn.functional import scaled_dot_product_attention
from .backbone_pointformer.pointformer_v3 import offset2bincount, offset2batch, batch2offset
from .embedder import RoPE3D, NerfPositionalEncoding, PositionEmbeddingCoordsSine

try:
    from xformers.ops import memory_efficient_attention, unbind, fmha

    print("xFormers is available.")
    XFORMERS_AVAILABLE = True
except ImportError:
    print("xFormers is not available.")
    XFORMERS_AVAILABLE = False


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_features": cfg.IN_FEATURES,
            "hidden_features": cfg.HIDDEN_FEATURES,
            "out_features": cfg.OUT_FEATURES,
            "act_layer": cfg.ACTIVATION,
            "bias": cfg.BIAS,
            "drop": cfg.DROP_OUT,
        }

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.DIM,
            "num_heads": cfg.NUM_HEADS,
            "qkv_bias": cfg.QKV_BIAS,
            "attn_drop": cfg.ATTN_DROP_OUT,
            "proj_drop": cfg.PROJ_DROP_OUT,
        }

    def forward(self, x, xpos):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        if self.rope is not None:
            q = self.rope(q, xpos.long())
            k = self.rope(k, xpos.long())

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.projq = nn.Linear(dim, dim, bias=qkv_bias)
        self.projk = nn.Linear(dim, dim, bias=qkv_bias)
        self.projv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    @classmethod
    def from_config(cls, cfg):
        return {
            "dim": cfg.DIM,
            "num_heads": cfg.NUM_HEADS,
            "qkv_bias": cfg.QKV_BIAS,
            "attn_drop": cfg.ATTN_DROP_OUT,
            "proj_drop": cfg.PROJ_DROP_OUT,
        }

    def forward(self, query, key, value, qpos, kpos):
        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q, qpos.long())
            k = self.rope(k, kpos.long())

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientAttention(Attention):
    def forward(self, x, xpos):
        if not XFORMERS_AVAILABLE:
            return super().forward(x, xpos)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]  # B x num_heads x N x C // num_heads

        if self.rope is not None:
            q = self.rope(q, xpos.long())
            k = self.rope(k, xpos.long())

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientAttentionV2(EfficientAttention):
    """
        Supports self attention of 2D image tokens
        and 3D point cloud tokens.

    """

    def __init__(
            self,
            dim,
            rope=None,
            num_heads=8,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.,
            pcd_patch_size=1024,
            order_index=0,
    ):

        super(EfficientAttentionV2, self).__init__(
            dim=dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )

        self.channels = dim
        self.patch_size = 0
        self.patch_size_max = pcd_patch_size
        self.order_index = order_index
        self.rope3d = RoPE3D(100)

    def forward_img_tokens(self, x, pos):
        return self.forward(x, pos)

    @torch.no_grad()
    def get_pos(self, point, order):
        pos_key = f"pos_{self.order_index}"
        if pos_key not in point.keys():
            point[pos_key] = point.grid_coord[order]
        return point[pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
                pad_key not in point.keys()
                or unpad_key not in point.keys()
                or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                    torch.div(
                        bincount + self.patch_size - 1,
                        self.patch_size,
                        rounding_mode="trunc",
                    )
                    * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]: _offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[
                    _offset_pad[i + 1]
                    - self.patch_size
                    + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                    ] = pad[
                        _offset_pad[i + 1]
                        - 2 * self.patch_size
                        + (bincount[i] % self.patch_size): _offset_pad[i + 1]
                                                           - self.patch_size
                        ]
                pad[_offset_pad[i]: _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward_pcd_tokens(self, x, point):
        self.patch_size = min(
            offset2bincount(point.offset).min().tolist(), self.patch_size_max
        )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(x)[order]

        q, k, v = (
            qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
        )

        pos = self.get_pos(point, order).reshape(-1, K, 3)

        # apply Rotary Position Embedding
        q = self.rope3d(q, pos.long())
        k = self.rope3d(k, pos.long())

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        feat = memory_efficient_attention(q, k, v)
        feat = feat.reshape(-1, C)

        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        return feat


class EfficientCrossAttention(CrossAttention):
    def forward(self, query, key, value, qpos, kpos):
        if not XFORMERS_AVAILABLE:
            return super().forward(query, key, value, qpos, kpos)

        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.projk(key).reshape(B, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.projv(value).reshape(B, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.rope is not None:
            q = self.rope(q, qpos.contiguous().long())
            k = self.rope(k, kpos.long())

        # (batch_size, seqlen, nheads, headdim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        x = memory_efficient_attention(q, k, v)
        x = x.reshape([B, Nq, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EfficientCrossAttentionV4(EfficientCrossAttention):
    """
        Cross Attention from 2D to 3D.
    """

    def __init__(self, dim, rope=None, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., order_index=0,
                 query_pos_embed_3d=None):
        super(EfficientCrossAttentionV4, self).__init__(
            dim, rope=rope, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.order_index = order_index
        self.rope3d = RoPE3D(100)  # make it configurable.

        if query_pos_embed_3d == 'nerf':
            self.point_query_pos_embed_3d = NerfPositionalEncoding(dim // 6)
        elif query_pos_embed_3d == "coords_sine":
            self.point_query_pos_embed_3d = PositionEmbeddingCoordsSine(n_dim=3, d_model=dim)

    def offset2batch(self, point, pos3d, offset):
        start = 0
        batch = []
        pos = []
        for stop in offset:
            batch.append(point[start:stop][None])
            pos.append(pos3d[start:stop][None])
            start = stop

        return batch, pos

    def batch2offset(self, batch):
        batch = [item.squeeze(0) for item in batch]
        return torch.vstack(batch)

    def forward_pcd_to_pcd(self, query, key, value, qpos, kpos, src_point_offset, tgt_point_offset):
        if not XFORMERS_AVAILABLE:
            raise NotImplementedError

        _, C = query.shape
        assert tgt_point_offset.shape[0] == src_point_offset.shape[0]
        query_batch, qpos_batch = self.offset2batch(query, qpos, src_point_offset)
        key_batch, kpos_batch = self.offset2batch(key, kpos, tgt_point_offset)
        val_batch, _ = self.offset2batch(value, kpos, tgt_point_offset)

        x = []
        for i in range(len(query_batch)):
            q_i, k_i, v_i = query_batch[i], key_batch[i], val_batch[i]

            Nq = q_i.shape[1]
            Nk = k_i.shape[1]
            Nv = v_i.shape[1]

            q_i = self.projq(q_i).reshape(1, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q_i = self.rope3d(q_i, qpos_batch[i].long())
            k_i = self.projk(k_i).reshape(1, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            k_i = self.rope3d(k_i, kpos_batch[i].long())
            v_i = self.projv(v_i).reshape(1, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            q_i = q_i.permute(0, 2, 1, 3)
            k_i = k_i.permute(0, 2, 1, 3)
            v_i = v_i.permute(0, 2, 1, 3)

            output = memory_efficient_attention(q_i, k_i, v_i)
            x.append(output.reshape(1, Nq, C))

        x = self.batch2offset(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_pcd_query_to_fused_feat(self, query, key, value, qpos, kpos, point_offset):
        if not XFORMERS_AVAILABLE:
            raise NotImplementedError

        B, Nq, C = query.shape
        Nk = key.shape[1]
        Nv = value.shape[1]

        # 3D point queries
        query += self.point_query_pos_embed_3d(qpos)
        q = self.projq(query).reshape(B, Nq, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        key_batch, kpos_batch = self.offset2batch(key, kpos, point_offset)
        val_batch, _ = self.offset2batch(value, kpos, point_offset)

        x = []
        for i in range(B):
            q_i, k_i, v_i = q[i][None], key_batch[i], val_batch[i]

            Nk = key_batch[i].shape[1]  # n x number of points x dim
            Nv = val_batch[i].shape[1]  # n x number of points x dim

            # point
            k_i = self.projk(k_i).reshape(1, Nk, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_i = self.projv(v_i).reshape(1, Nv, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            k_i = self.rope3d(k_i, kpos_batch[i].long())

            q_i = q_i.permute(0, 2, 1, 3)
            k_i = k_i.permute(0, 2, 1, 3)
            v_i = v_i.permute(0, 2, 1, 3)

            x.append(memory_efficient_attention(q_i, k_i, v_i))

        x = torch.vstack(x)
        x = x.reshape([B, Nq, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlockV4(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, norm_mem=True, rope=None, pcd_patch_size=1024, order_index=0,
                 use_flash_attn=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if use_flash_attn:
            raise NotImplementedError
        else:
            self.attn = EfficientAttentionV2(
                dim,
                rope=rope,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                pcd_patch_size=pcd_patch_size,
                order_index=order_index,
            )

            self.cross_attn = EfficientCrossAttentionV4(
                dim,
                rope=rope,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                order_index=order_index
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        self.norm_y = norm_layer(dim) if norm_mem else nn.Identity()
        self.order_index = order_index

    def forward_pcd_to_pcd(self, src, tgt, src_point, tgt_point):
        src_order = src_point.serialized_order[self.order_index]
        tgt_order = tgt_point.serialized_order[self.order_index]
        src_inverse = src_point.serialized_inverse[self.order_index]
        tgt_inverse = tgt_point.serialized_inverse[self.order_index]
        src = src[src_order]
        tgt = tgt[tgt_order]
        src_pos = src_point.grid_coord[src_order]
        tgt_pos = tgt_point.grid_coord[tgt_order]

        src = src + self.drop_path(self.attn.forward_pcd_tokens(self.norm1(src), src_point))
        tgt = self.norm_y(tgt)
        src = src + self.drop_path(
            self.cross_attn.forward_pcd_to_pcd(self.norm2(src), tgt, tgt, src_pos, tgt_pos, src_point.offset,
                                               tgt_point.offset)[src_inverse])

        tgt = tgt[tgt_inverse]
        src = src + self.drop_path(self.mlp(self.norm3(src)))
        return src, tgt


class QueryDecoderBlockV2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, norm_mem=True, rope=None, order_index=0, query_pos_embed_3d='nerf'):
        super().__init__()

        self.is_rope_pos_enc = False
        if rope is not None:
            self.is_rop_pos_enc = True

        self.norm1 = norm_layer(dim)

        self.cross_attn = EfficientCrossAttentionV4(
            dim,
            rope=rope,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            order_index=order_index,
            query_pos_embed_3d=query_pos_embed_3d
        )

        # Feed forward
        ff_hidden_dim = int(dim * mlp_ratio)
        self.linear1 = nn.Linear(dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, dim)
        self.act = nn.GELU()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = nn.Dropout(drop)
        self.dropout3 = nn.Dropout(drop)

        self.norm_mem = norm_layer(dim) if norm_mem else nn.Identity()
        self.order_index = order_index

    def with_pos_embed(self, tensor, pos: Optional[torch.Tensor]):
        # print(self.is_rope_pos_enc, tensor.shape, pos.shape)
        if self.is_rope_pos_enc:
            return tensor

        return tensor if pos is None else tensor + pos

    def forward_pcd_to_img(self, tgt, memory, point, queries_emb, query_pos):
        memory = self.norm_mem(memory)

        # point serialization order
        order = point.serialized_order[self.order_index]
        memory = memory[order]
        kpos = point.grid_coord[order]

        tgt2 = self.cross_attn.forward_pcd_query_to_fused_feat(
            query=self.with_pos_embed(tgt, queries_emb),
            key=memory,
            value=memory,
            qpos=query_pos,
            kpos=kpos,
            point_offset=point.offset
        )

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout2(self.act(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt
