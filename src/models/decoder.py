# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).


# --------------------------------------------------------
# References: https://github.com/naver/croco/blob/master/models/croco.py
# --------------------------------------------------------

import torch
import torch.nn as nn

from functools import partial

from .blocks import DecoderBlockV4, QueryDecoderBlockV2, Mlp


class MaPoDecoderV10(nn.Module):
    """ Adapted from Croco V2

        This decoder fuses 2d-3d point cloud features.
        Decoder output consists of correspondence and uncertainty weights.
    """

    def __init__(self,
                 pos_embed,
                 enc_embed_dim=1024,
                 enc_embed_dim_3d=768,
                 dec_embed_dim=864,
                 depth=8,
                 num_heads=16,
                 mlp_ratio=4,
                 point_order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 point_patch_size=1024,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_img2=True,  # whether to apply normalization of the 'memory' = (second image) in the decoder
                 rope=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.,
                 pred_dim_3d=4,
                 pcd_patch_size=1024,
                 query_pos_embed_3d="nerf",
                 use_flash_attn=False,
                 **kwargs
                 ):
        super(MaPoDecoderV10, self).__init__()
        # assert pos_embed is not None or rope is not None, "No position embedding provided."

        if pos_embed is not None:
            self.register_buffer('dec_pos_embed', torch.from_numpy(pos_embed).float())
        else:
            self.dec_pos_embed = None

        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)

        # point
        self.point_order = point_order
        self.point_patch_size = point_patch_size

        self.order_indices = [i % len(point_order) for i in range(depth)]

        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlockV4(
                dim=dec_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                norm_mem=norm_img2,
                rope=rope,
                drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                pcd_patch_size=pcd_patch_size,
                order_index=self.order_indices[i],
                use_flash_attn=use_flash_attn
            ) for i in range(depth)])

        # query projector and decoder
        self.project_dim = dec_embed_dim // 2
        self.mem_proj = nn.Linear(dec_embed_dim, self.project_dim)
        self.dec_query_blocks = nn.ModuleList([
            QueryDecoderBlockV2(
                dim=self.project_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                norm_mem=norm_img2,
                rope=rope,
                drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                order_index=self.order_indices[i],
                query_pos_embed_3d=query_pos_embed_3d
            ) for i in range(depth)])

        # final norm layer
        self.dec_norm1 = norm_layer(dec_embed_dim)
        self.dec_norm2 = norm_layer(self.project_dim)

        # 3d correspondece embed (N x num_queries x 4)
        # corrs(3) + confidence(1)
        self.corr_embed_3d = Mlp(self.project_dim, hidden_features=self.project_dim, out_features=pred_dim_3d)

    @classmethod
    def from_config(cls, cfg):
        instance = cls(pos_embed=cfg.pos_embed,
                       enc_embed_dim=cfg.enc_embed_dim,
                       enc_embed_dim_3d=cfg.enc_embed_dim_3d,
                       dec_embed_dim=cfg.dec_embed_dim,
                       depth=cfg.depth,
                       num_heads=cfg.num_heads,
                       mlp_ratio=cfg.mlp_ratio,
                       point_patch_size=cfg.point_patch_size,
                       qkv_bias=cfg.qkv_bias,
                       norm_img2=cfg.norm_img2,
                       rope=cfg.rope,
                       attn_drop=cfg.attn_drop,
                       proj_drop=cfg.proj_drop,
                       drop_path=cfg.drop_path,
                       pred_dim_3d=cfg.pred_dim_3d,
                       pcd_patch_size=cfg.pcd_patch_size,
                       query_pos_embed_3d=cfg.query_pos_embed_3d,
                       use_flash_attn=cfg.use_flash_attn)
        return instance

    def forward_pcd_to_pcd(self, src_point, tgt_point, query_pos, **kwargs):
        assert query_pos.shape[-1] == 3, f"Invalid queries dim. Expected 3 but recieved {query_pos.shape[-1]}"

        src_feat = self.decoder_embed(src_point.feat)
        tgt_feat = self.decoder_embed(tgt_point.feat)

        # COTR query (not ideal)
        _b, _q, _ = query_pos.shape
        q = torch.zeros(_b, _q, self.project_dim).to(src_feat.device)

        tmp1 = src_feat
        tmp2 = tgt_feat
        for idx, (block1, block2) in enumerate(zip(self.dec_blocks, self.dec_query_blocks)):
            tmp1, tmp2 = block1.forward_pcd_to_pcd(tmp1, tmp2, src_point, tgt_point)
            mem = self.mem_proj(tmp1)
            q = block2.forward_pcd_to_img(q, mem, src_point, None, query_pos)

        out = self.dec_norm1(tmp1)
        q = self.dec_norm2(q)

        output = self.corr_embed_3d(q)
        corr = output[..., :3]
        info = output[..., 3:]

        return corr, info, out
