"""REGTR network architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

from .backbone_pointformer.pointformer_encoder import PTv3_Encoder
from .generic_reg_model import GenericRegModel
from .losses.corr_loss import CorrCriterion
from .losses.feature_loss import InfoNCELossFull, CircleLossFull
from .transformer.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from .transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_transform, se3_inv, \
    compute_rigid_transform_with_sinkhorn, pairwise_distance
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
from .blocks import QueryDecoderBlockV2, Mlp

_TIMEIT = False
import torch.utils.checkpoint

"""
This implementation contains every test
1. Lowe's ratio test
2. LGR
3. Ransac
4. Using Overlap values as weights
"""


def offset2lengths(offsets):
    lengths = torch.cat([offsets[:1], offsets[1:] - offsets[:-1]])
    return lengths

def concat_batch(batch):
    sample = torch.concat([_.squeeze(0) for _ in batch], dim=0)
    return sample

def compute_overlaps(batch):
    """Compute groundtruth overlap for each point+level. Note that this is a
    approximation since
    1) it relies on the pooling indices from the preprocessing which caps the number of
       points considered
    2) we do a unweighted average at each level, without considering the
       number of points used to generate the estimate at the previous level
    """

    src_overlap = batch['src_overlap']
    tgt_overlap = batch['tgt_overlap']
    kpconv_meta = batch['kpconv_meta']
    n_pyr = len(kpconv_meta['points'])

    overlap_pyr = {'pyr_0': torch.cat(src_overlap + tgt_overlap, dim=0).type(torch.float)}
    invalid_indices = [s.sum() for s in kpconv_meta['stack_lengths']]
    for p in range(1, n_pyr):
        pooling_indices = kpconv_meta['pools'][p - 1].clone()
        valid_mask = pooling_indices < invalid_indices[p - 1]
        pooling_indices[~valid_mask] = 0

        # Average pool over indices
        overlap_gathered = overlap_pyr[f'pyr_{p - 1}'][pooling_indices] * valid_mask
        overlap_gathered = torch.sum(overlap_gathered, dim=1) / torch.sum(valid_mask, dim=1)
        overlap_gathered = torch.clamp(overlap_gathered, min=0, max=1)
        overlap_pyr[f'pyr_{p}'] = overlap_gathered

    return overlap_pyr


class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.cfg = cfg

        #######################
        # KPConv Encoder/decoder
        #######################
        self.point_encoder = PTv3_Encoder(cfg, cfg.d_embed)

        #######################
        # Embeddings
        #######################
        if cfg.get('pos_emb_type', 'sine') == 'sine':
            self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.d_embed,
                                                         scale=cfg.get('pos_emb_scaling', 1.0))
        elif cfg['pos_emb_type'] == 'learned':
            self.pos_embed = PositionEmbeddingLearned(3, cfg.d_embed)
        else:
            raise NotImplementedError

        #######################
        # Attention propagation
        #######################
        encoder_layer = TransformerCrossEncoderLayer(
            cfg.d_embed, cfg.nhead, cfg.d_feedforward, cfg.dropout,
            activation=cfg.transformer_act,
            normalize_before=cfg.pre_norm,
            sa_val_has_pos_emb=cfg.sa_val_has_pos_emb,
            ca_val_has_pos_emb=cfg.ca_val_has_pos_emb,
            attention_type=cfg.attention_type,
        )
        encoder_norm = nn.LayerNorm(cfg.d_embed) if cfg.pre_norm else None
        self.transformer_encoder = TransformerCrossEncoder(
            encoder_layer, cfg.num_encoder_layers, encoder_norm,
            return_intermediate=False)

        self.softplus = torch.nn.Softplus()

        self.order_indices = [i % len(cfg.point_order) for i in range(cfg.depth)]
        self.dec_query_blocks = nn.ModuleList([
            QueryDecoderBlockV2(
                dim=cfg.d_embed,
                num_heads=cfg.num_heads,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                norm_mem=cfg.norm_mem,
                rope=cfg.rope,
                drop=cfg.proj_drop,
                attn_drop=cfg.attn_drop,
                drop_path=cfg.drop_path,
                order_index=self.order_indices[i],
                query_pos_embed_3d=cfg.query_pos_embed_3d
            ) for i in range(cfg.depth)])
        self.dec_norm = nn.LayerNorm(cfg.d_embed, eps=1e-6)

        self.corr_embed_3d = Mlp(cfg.project_dim, hidden_features=cfg.project_dim, out_features=4)

    def forward(self, batch):
        src_pcd = {
            "feat": batch["src_pcd"],
            "coord": batch["src_pcd"],
            "grid_coord": batch["src_grid_coord"].int(),
            "offset": torch.cumsum(batch["src_length"], dim=0).to(batch["src_pcd"].device)
        }

        tgt_pcd = {
            "feat": batch["tgt_pcd"],
            "coord": batch["tgt_pcd"],
            "grid_coord": batch["tgt_grid_coord"].int(),
            "offset": torch.cumsum(batch["tgt_length"], dim=0).to(batch["tgt_pcd"].device)
        }

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        src_feat = self.point_encoder(src_pcd)
        tgt_feat = self.point_encoder(tgt_pcd)
        src_lens = offset2lengths(src_feat['offset']).tolist()
        tgt_lens = offset2lengths(tgt_feat['offset']).tolist()

        src_feats = torch.split(src_feat['feat'], src_lens, dim=0)
        tgt_feats = torch.split(tgt_feat['feat'], tgt_lens, dim=0)

        ##### NEED CHECK
        src_pe = torch.split(self.pos_embed(src_feat['coord']), src_lens, dim=0)
        tgt_pe = torch.split(self.pos_embed(tgt_feat['coord']), tgt_lens, dim=0)
        src_pe_padded, _, _ = pad_sequence(src_pe)
        tgt_pe_padded, _, _ = pad_sequence(tgt_pe)

        # Performs padding, then apply attention (REGTR "encoder" stage) to condition on the other
        # point cloud

        src_feats_padded, src_key_padding_mask, _ = pad_sequence(src_feats,
                                                                 require_padding_mask=True)
        tgt_feats_padded, tgt_key_padding_mask, _ = pad_sequence(tgt_feats,
                                                                 require_padding_mask=True)
        src_feats_cond, tgt_feats_cond = self.transformer_encoder(
            src_feats_padded, tgt_feats_padded,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            src_pos=src_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
            tgt_pos=tgt_pe_padded if self.cfg.transformer_encoder_has_pos_emb else None,
        )
        # CHECK FORMAT
        src_feats_cond_unpad = unpad_sequences(src_feats_cond, src_lens)
        src_feats_cond = concat_batch(src_feats_cond_unpad)
        tgt_feats_cond_unpad = unpad_sequences(tgt_feats_cond, tgt_lens)
        tgt_feats_cond = concat_batch(tgt_feats_cond_unpad)

        ###########################
        # Correspondence Prediction
        ###########################
        queries = batch['queries']
        _b, _q, _ = queries.shape
        q = torch.zeros(_b, _q, self.cfg.project_dim).to(src_feats_cond.device)

        for block in self.dec_query_blocks:
            q = block.forward_pcd_to_img(q, tgt_feats_cond, src_feat, None, queries)
        q = self.dec_norm(q)
        output = self.corr_embed_3d(q)
        corr = output[..., :3]
        info = output[..., 3:]

        outputs = {
            # Predictions
            'src_feat': src_feats_cond,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_cond,  # List(B) of (N_pred, N_tgt, D)

            'norm_corr': corr,
            'conf_info': info,
        }
        return outputs

    def compute_loss(self, pred, batch):
        loss_details = {}
        total_loss = 0

        # Feature Loss
        if self.cfg.feature_loss:
            raise NotImplementedError

        norm_targets = batch['norm_targets']
        norm_corr = pred['norm_corr']
        if self.cfg.reg_loss == "l1":
            loss = F.l1_loss(norm_corr, norm_targets, reduction="none").sum(dim=2, keepdim=True)
        elif self.cfg.reg_loss == "l2":
            loss = torch.norm(norm_targets - norm_corr, dim=2, keepdim=True)
        elif self.cfg.reg_loss == "smooth_l1":
            loss = F.smooth_l1_loss(norm_corr, norm_targets, reduction="none").sum(dim=2, keepdim=True)
        else:
            raise ValueError(f"Unsupported {self.cfg.reg_loss}. ")
        loss_details['regression'] = loss

        conf = pred['conf_info']
        vmin = float(self.cfg.vmin)
        vmax = float(self.cfg.vmax)
        if self.cfg.mode == 'exp':
            conf = vmin + conf.exp().clip(max=vmax - vmin)
        elif self.cfg.mode == 'sigmoid':
            conf = (vmax - vmin) * torch.sigmoid(conf) + vmin
        else:
            raise ValueError(f"Unsupported {self.cfg.mode}. ")

        log_conf = torch.log(conf)
        conf_loss = loss * conf - self.alpha * log_conf
        loss_details['confidence_loss'] = conf_loss
        total_loss += conf_loss.mean() if conf_loss.numel() > 0 else 0
        loss_details['total'] = total_loss

        return loss_details

    def recompute_weights(self, src_points, tgt_points, weights, pose):
        src_points_tf = se3_transform(pose, src_points)
        residuals = torch.linalg.norm(tgt_points - src_points_tf, dim=1)
        inlier_masks = torch.lt(residuals, self.cfg.acceptance_radius)
        new_weights = weights * inlier_masks.float()
        return new_weights

    def local_global_registration(self, src_points, tgt_points, weights, pose):
        for _ in range(self.cfg.num_refinement_steps):
            weights = self.recompute_weights(src_points, tgt_points, weights, pose)
            pose = compute_rigid_transform(src_points, tgt_points, weights)

        return pose
