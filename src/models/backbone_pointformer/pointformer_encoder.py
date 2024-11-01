import torch.nn as nn

from .pointformer_v3 import PointTransformerV3


class PTv3_Encoder(PointTransformerV3):
    """
    Point Transformer V3 Encoder Wrapper class.

    """

    def __init__(self, cfg, project_dim,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 upcast_attention=False,
                 upcast_softmax=False,
                 cls_mode=True,
                 pdnorm_bn=False,
                 pdnorm_ln=False,
                 pdnorm_decouple=True,
                 pdnorm_adaptive=False,
                 pdnorm_affine=True,
                 pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")
                 ):
        super(PTv3_Encoder, self).__init__(
            in_channels=cfg.in_channels,
            order=order,
            stride=cfg.stride,
            enc_depths=cfg.enc_depths,
            enc_channels=cfg.enc_channels,
            enc_num_head=cfg.enc_num_head,
            enc_patch_size=cfg.enc_patch_size,
            mlp_ratio=cfg.mlp_ratio,
            qkv_bias=cfg.qkv_bias,
            qk_scale=cfg.qk_scale,
            attn_drop=cfg.attn_drop,
            proj_drop=cfg.proj_drop,
            drop_path=cfg.drop_path,
            pre_norm=cfg.pre_norm,
            shuffle_orders=cfg.shuffle_orders,
            enable_rpe=cfg.enable_rpe,
            enable_flash=cfg.enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            cls_mode=cls_mode,
            pdnorm_bn=pdnorm_bn,
            pdnorm_ln=pdnorm_ln,
            pdnorm_decouple=pdnorm_decouple,
            pdnorm_adaptive=pdnorm_adaptive,
            pdnorm_affine=pdnorm_affine,
            pdnorm_conditions=pdnorm_conditions
        )

        self.project = nn.Identity()
        if project_dim:
            self.project = nn.Linear(cfg.enc_channels[-1], project_dim, bias=True)

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.IN_CHANNELS,
            "stride": cfg.STRIDE,
            "enc_depths": cfg.ENC_DEPTHS,
            "enc_channels": cfg.ENC_CHANNELS,
            "enc_num_head": cfg.ENC_NUM_HEADS,
            "enc_patch_size": cfg.ENC_PATCH_SIZE,
            "mlp_ratio": cfg.MLP_RATIO,
            "qkv_bias": cfg.QKV_BIAS,
            "qk_scale": cfg.QK_SCALE,
            "attn_drop": cfg.ATTN_DROP,
            "proj_drop": cfg.PROJ_DROP,
            "drop_path": cfg.DROP_PATH,
            "pre_norm": cfg.PRE_NORM,
            "shuffle_orders": cfg.SHUFFLE_ORDERS,
            "enable_rpe": cfg.ENABLE_RPE,
            "enable_flash": cfg.ENABLE_FLASH,
            "project_dim": cfg.PROJECT_DIM
        }

    def forward(self, pcd, **kwargs):
        output = super().forward(pcd)

        output.feat = self.project(output.feat)
        return output
