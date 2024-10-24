import torch.nn as nn

from .pointformer_v3 import PointTransformerV3


class PTv3_Encoder(PointTransformerV3):
    """
    Point Transformer V3 Encoder Wrapper class.

    """

    def __init__(self,
                 in_channels=3,
                 order=("z", "z-trans", "hilbert", "hilbert-trans"),
                 stride=(2, 2, 2, 2),
                 enc_depths=(2, 2, 2, 6, 2),
                 enc_channels=(32, 64, 128, 256, 512),
                 enc_num_head=(2, 4, 8, 16, 32),
                 enc_patch_size=(1024, 1024, 1024, 1024, 1024),
                 mlp_ratio=4,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 drop_path=0.3,
                 pre_norm=True,
                 shuffle_orders=True,
                 enable_rpe=False,
                 enable_flash=False,
                 upcast_attention=False,
                 upcast_softmax=False,
                 cls_mode=True,
                 pdnorm_bn=False,
                 pdnorm_ln=False,
                 pdnorm_decouple=True,
                 pdnorm_adaptive=False,
                 pdnorm_affine=True,
                 project_dim=None,
                 pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D")
                 ):
        super(PTv3_Encoder, self).__init__(
            in_channels=in_channels,
            order=order,
            stride=stride,
            enc_depths=enc_depths,
            enc_channels=enc_channels,
            enc_num_head=enc_num_head,
            enc_patch_size=enc_patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            drop_path=drop_path,
            pre_norm=pre_norm,
            shuffle_orders=shuffle_orders,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
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
            self.project = nn.Linear(enc_channels[-1], project_dim, bias=True)

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
