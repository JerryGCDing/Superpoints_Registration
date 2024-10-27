"""REGTR network architecture
"""
import math
import time
import torch
import torch.nn as nn

from .backbone_pointformer.pointformer_encoder import PTv3_Encoder
from .backbone_kpconv.kpconv import compute_overlaps
from .generic_reg_model import GenericRegModel
from .losses.corr_loss import CorrCriterion
from .losses.feature_loss import InfoNCELossFull, CircleLossFull
from .transformer.position_embedding import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from .transformer.transformers import TransformerCrossEncoderLayer, TransformerCrossEncoder
from utils.se3_torch import compute_rigid_transform, se3_transform_list, se3_transform, se3_inv, compute_rigid_transform_with_sinkhorn, pairwise_distance
from utils.seq_manipulation import split_src_tgt, pad_sequence, unpad_sequences
_TIMEIT = False
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.utils.checkpoint
"""
This implementation contains every test
1. Lowe's ratio test
2. LGR
3. Ransac
4. Using Overlap values as weights
"""

class RegTR(GenericRegModel):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.verbose = False
        self.cfg = cfg

        #######################
        # KPConv Encoder/decoder
        #######################
        self.kpf_encoder = PTv3_Encoder(cfg, cfg.d_embed)
        # Bottleneck layer to shrink KPConv features to a smaller dimension for running attention
        self.feat_proj = nn.Linear(self.kpf_encoder.encoder_skip_dims[-1], cfg.d_embed, bias=True)

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

        # Affinity parameters
        self.beta = torch.nn.Parameter(torch.tensor(1.0))
        self.alpha = torch.nn.Parameter(torch.tensor(1.0))

        self.softplus = torch.nn.Softplus()

        ############################
        # Overlap Prediction layers
        ############################
        self.overlap_predictor = nn.Linear(cfg.d_embed, 1)

        #######################
        # Losses
        #######################
        self.overlap_criterion = nn.BCEWithLogitsLoss()
        if self.cfg.feature_loss_type == 'infonce':
            self.feature_criterion = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = InfoNCELossFull(cfg.d_embed, r_p=cfg.r_p, r_n=cfg.r_n)
        elif self.cfg.feature_loss_type == 'circle':
            self.feature_criterion = CircleLossFull(dist_type='euclidean', r_p=cfg.r_p, r_n=cfg.r_n)
            self.feature_criterion_un = self.feature_criterion
        else:
            raise NotImplementedError

        self.corr_criterion = CorrCriterion(metric='mae')

        self.weight_dict = {}
        for k in ['overlap', 'feature', 'corr']:
            for i in cfg.get(f'{k}_loss_on', [cfg.num_encoder_layers - 1]):
                self.weight_dict[f'{k}_{i}'] = cfg.get(f'wt_{k}')
        self.weight_dict['feature_un'] = cfg.wt_feature_un

        self.logger.info('Loss weighting: {}'.format(self.weight_dict))
        self.logger.info(
            f'Config: d_embed:{cfg.d_embed}, nheads:{cfg.nhead}, pre_norm:{cfg.pre_norm}, '
            f'use_pos_emb:{cfg.transformer_encoder_has_pos_emb}, '
            f'sa_val_has_pos_emb:{cfg.sa_val_has_pos_emb}, '
            f'ca_val_has_pos_emb:{cfg.ca_val_has_pos_emb}'
        )
        self.weight_matrix = []
        self.conf_matrix = []
        self.pos_class = []
        self.neg_class = []

        self.dual_normalization = True
        self.num_points_M = []
        self.num_points_N = []
        self.num_points_MP = []
        self.num_points_NP = []

    def forward(self, batch):
        src_pcd = {
            "feat": batch["src_points"][0],
            "coord": batch["src_points"][0],
            "grid_coord": batch["src_grid"].int(),
            "offset": torch.cumsum(batch["src_length"][0], dim=0).to(batch["src_points"][0].device)
        }
        src_lens = batch["src_length"]

        tgt_pcd = {
            "feat": batch["tgt_points"][0],
            "coord": batch["tgt_points"][0],
            "grid_coord": batch["tgt_grid"].int(),
            "offset": torch.cumsum(batch["tgt_length"][0], dim=0).to(batch["tgt_points"][0].device)
        }
        tgt_lens = batch["tgt_length"]

        ####################
        # REGTR Encoder
        ####################
        # KPConv encoder (downsampling) to obtain unconditioned features
        src_feat = self.point_encoder(src_pcd)
        tgt_feat = self.point_encoder(tgt_pcd)

        src_feat = self.feat_proj(src_feat)
        src_feats = torch.split(src_feat, src_lens, dim=0)
        tgt_feat = self.feat_proj(tgt_feat)
        tgt_feats = torch.split(tgt_feat, tgt_lens, dim=0)

        ##### NEED CHECK
        print(src_pcd['coord'].shape)
        src_pe = torch.split(self.pos_embed(src_pcd['coord']), src_lens, dim=0)
        tgt_pe = torch.split(self.pos_embed(tgt_pcd['coord']), tgt_lens, dim=0)
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
        
        #####################
        # Overlap Prediction
        #####################
        src_overlap = torch.sigmoid(self.overlap_predictor(src_feats_cond))
        tgt_overlap = torch.sigmoid(self.overlap_predictor(tgt_feats_cond))

        src_overlap_list = unpad_sequences(src_overlap, src_lens)
        tgt_overlap_list = unpad_sequences(tgt_overlap, tgt_lens)

        #####################
        src_feats_cond_unpad = unpad_sequences(src_feats_cond, src_lens)
        tgt_feats_cond_unpad = unpad_sequences(tgt_feats_cond, tgt_lens)

        # Softmax Correlation
        src_coords = torch.split(src_dict['coord'], src_lens, dim=0)
        tgt_coords = torch.split(tgt_dict['coord'], tgt_lens, dim=0)
        pose_sfc, attn_list, overlap_prob_list, ind_list, src_pts_list, tgt_pts_list = self.softmax_correlation(src_feats_cond_unpad, tgt_feats_cond_unpad,
                                     src_coords, tgt_coords, src_overlap_list, tgt_overlap_list)

        outputs = {
            # Predictions
            'pose': pose_sfc,
            'attn': attn_list,
            'src_feat': src_feats_cond_unpad,  # List(B) of (N_pred, N_src, D)
            'tgt_feat': tgt_feats_cond_unpad,  # List(B) of (N_pred, N_tgt, D)

            'src_kp': src_coords,
            'tgt_kp': tgt_coords,

            'src_corr': src_pts_list,
            'tgt_corr': tgt_pts_list,

            'src_overlap': src_overlap_list,
            'tgt_overlap': tgt_overlap_list,

            'overlap_prob_list': overlap_prob_list,
            'ind_list': ind_list,
        }
        return outputs

    def compute_loss(self, pred, batch):
        losses = {}
        kpconv_meta = batch['kpconv_meta']
        pose_gt = batch['pose']
        p = len(kpconv_meta['stack_lengths']) - 1 

        try:
            batch['overlap_pyr'] = compute_overlaps(batch)
            src_overlap_p, tgt_overlap_p = \
                split_src_tgt(batch['overlap_pyr'][f'pyr_{p}'], kpconv_meta['stack_lengths'][p])

            # Overlap prediction loss
            all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
            all_overlap_gt = batch['overlap_pyr'][f'pyr_{p}']
            
            overlap_loss = self.overlap_criterion(all_overlap_pred[0, :, 0], all_overlap_gt)
        except:
            print("Error in overlap loss")
        
        # Inlier Loss
        if self.cfg.inlier_loss_on:
            inlier_loss = 0
            for i in range(len(batch['pose'])):
                inlier_loss += torch.linalg.norm(pred['tgt_corr'][i] - se3_transform(pred['pose'][i], pred['src_corr'][i]), dim=1).mean()
        
        # Feature Loss
        for i in self.cfg.feature_loss_on:
            feature_loss = self.feature_criterion(
                [s[i] for s in pred['src_feat']],
                [t[i] for t in pred['tgt_feat']],
                se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp'],
            )

        # Transformation Loss
        pc_tf_gt = se3_transform_list(pose_gt, pred['src_kp'])
        pc_tf_pred = se3_transform_list(pred['pose'], pred['src_kp'])

        T_loss = 0
        for i in range(len(pc_tf_gt)):
            T_loss += torch.mean(torch.abs(pc_tf_gt[i] - pc_tf_pred[i])).requires_grad_()
        
        if self.verbose:
            print(f"Feature loss: {feature_loss}")
            print(f"Overlap loss: {overlap_loss}")
            print(f"T loss: {T_loss}")

        losses['feature'] = feature_loss
        losses['T'] = T_loss
        losses['overlap'] = overlap_loss
        losses['total'] = T_loss + 0.1 * feature_loss + overlap_loss 
        
        if self.cfg.inlier_loss_on:
            losses['inlier'] = inlier_loss
            losses['total'] += inlier_loss
        return losses

    def ratio_test(self, attn, dim):
        if dim==1:
            val2, ind2 = torch.topk(attn,2, dim=1)
            val2_ratio = val2[:,1,:]/val2[:,0,:]
            val2 = torch.where(val2_ratio<self.cfg.lowe_thres, val2, 0)
            val = val2[:,0,:]
            ind = ind2[:,0,:]
        elif dim==2:
            val2, ind2 = torch.topk(attn,2, dim=2)
            val2_ratio = val2[:,:,1]/val2[:,:,0]
            val2 = torch.where(val2_ratio<self.cfg.lowe_thres, val2[:,:,0], 0)
            val = val2
            ind = ind2[:,:,0]

        return ind, val

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
    
    def ransac(self, src, tgt, weights):
        N = src.size(0)
        itr = 500
        sample_size = 100
        for i in range(itr):
            idx = torch.randint(0, N, (sample_size,)).cuda()
            src_samples = torch.gather(src, 0, idx.unsqueeze(-1).expand(-1,3))
            tgt_samples = torch.gather(tgt, 0, idx.unsqueeze(-1).expand(-1,3))
            weight_samples = torch.gather(weights, 0, idx)

            T_estimated = compute_rigid_transform(src_samples, tgt_samples, weight_samples)
            src_tf = se3_transform(T_estimated, src)
            loss = torch.linalg.norm(tgt - src_tf, dim=1).mean()

            if i==0:
                T = T_estimated
                best_loss = loss
            elif loss<best_loss:
                T = T_estimated
                best_loss = loss

        return T 
    
    def softmax_correlation(self, src_feats, tgt_feats, src_xyz, tgt_xyz, src_overlap_list, tgt_overlap_list):
        """
        Args:
            src_feats_padded: Source features List of [1, N, D] tensors
            tgt_feats_padded: Target features List of [1, M, D] tensors
            src_xyz: List of ([N, 3])
            tgt_xyz: List of ([M, 3])
            src_overlap_list: List of ([N, 1])
            tgt_overlap_list: List of ([M, 1])
        Returns:
        """

        B = len(src_feats)
        pose_list = []
        attn_list = []

        # Variables to calculate overlap loss if overlap is calculated on correlation values
        overlap_prob_list = []
        ind_list = []
        src_pts_list = []
        tgt_pts_list = []

        for i in range(B):
            _, N, D = src_feats[i].shape
            _, M, D = tgt_feats[i].shape

            self.num_points_NP.append(N)
            self.num_points_MP.append(M)

            # Correlation = [1, N, M]
            correlation = torch.matmul(src_feats[i], tgt_feats[i].permute(0, 2, 1)) / (D**0.5)
            
            if N>M:
                if self.dual_normalization:
                    attn_src = torch.nn.functional.softmax(correlation, dim=-2)
                    attn_tgt = torch.nn.functional.softmax(correlation, dim=-1)
                    attn = attn_src * attn_tgt
                else:
                    attn = torch.nn.functional.softmax(correlation, dim=-2)
                # attn = torch.nn.functional.softmax(correlation, dim=-2)
                attn_list.append(attn)

                if self.cfg.use_ratio_test:
                    ind, val = self.ratio_test(attn, dim=1)
                else:
                    val, ind = torch.max(attn, dim=1)

                # Threshold the coorelation values?
                if self.cfg.threshold_corr:
                    # val = torch.where(val>self.cfg.corr_threshold, val, 0)
                    val = torch.where(val>torch.median(val), val, 0)

                if self.cfg.use_sinkhorn:
                    src_pts = src_xyz[i]
                else:
                    try:
                        src_pts = torch.gather(src_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [N, 3] -> [M, 3]
                    except:
                        raise ValueError("You found the error")
                tgt_pts = tgt_xyz[i]

                if self.cfg.remove_outliers_overlap:
                    src_overlap_prob = src_overlap_list[i].squeeze(2)
                    tgt_overlap_prob = tgt_overlap_list[i].squeeze(2)

                    src_overlap_prob = torch.gather(src_overlap_prob, 1, ind)
                    overlap_prob = src_overlap_prob * tgt_overlap_prob

                    if not self.cfg.use_overlap_as_weights:
                        # overlap_mask = overlap_prob > self.cfg.overlap_threshold
                        # val = torch.where(overlap_mask, val, 0)
                        val = val * overlap_prob

                # print(f"src_pts shape is: {src_pts.shape}")
                # print(f"tgt_pts shape is: {tgt_pts.shape}")
                # print(f"new N is: {int(0.2*M)}")
                if self.cfg.remove_points_from_val:
                    val, ind = torch.topk(val, int(self.cfg.val_threshold*M), dim=1)
                    src_pts = torch.gather(src_pts, 0, ind.permute(1,0).expand(-1,3))
                    tgt_pts = torch.gather(tgt_pts, 0, ind.permute(1,0).expand(-1,3))

                # print(f"src_pts shape is: {src_pts.shape}")
                # print(f"tgt_pts shape is: {tgt_pts.shape}")
                
                # raise ValueError

                # Compute the transformation matrix
                if self.cfg.use_sinkhorn:
                    if self.cfg.use_attn_affinity: # Use attention matrix as the affinity matrix
                        affinity = torch.gather(attn, 1, ind.unsqueeze(-1).expand(-1,-1,M))
                        print(src_pts.shape)
                        print(tgt_xyz[i].shape)
                        print(tgt_pts.shape)
                        raise ValueError
                        T = compute_rigid_transform_with_sinkhorn(src_pts, tgt_xyz[i], affinity, self.cfg.slack, self.cfg.sinkhorn_itr)

                    elif self.cfg.use_corr_affinity: # Compute affinity matrix from the correlation matrix
                        corr = torch.gather(correlation, 1, ind.unsqueeze(-1).expand(-1,-1,M))
                        score_matrix = torch.clamp(1-corr, min=0.0, max=None)
                        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
                        T = compute_rigid_transform_with_sinkhorn(src_pts, tgt_xyz[i], affinity, self.cfg.slack, self.cfg.sinkhorn_itr)
                    
                    else:
                        # score_matrix = pairwise_distance(src_feats[i], tgt_feats[i])
                        # print(score_matrix.shape)
                        # print(score_matrix.min())
                        # print(score_matrix.max())
                        # raise ValueError

                        score_matrix = torch.matmul(src_feats[i], tgt_feats[i].permute(0, 2, 1)) / (D**0.5)
                        score_matrix = torch.clamp(score_matrix, min=0.0, max=None)
                        
                        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
                        T = compute_rigid_transform_with_sinkhorn(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0), affinity, self.cfg.slack, self.cfg.sinkhorn_itr)


                        # score_matrix = torch.clamp(1-correlation, min=0.0, max=None)
                        # affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
                        # print(affinity.shape)
                        # T = compute_rigid_transform_with_sinkhorn_2(src_xyz[i], tgt_xyz[i], affinity, self.cfg.slack, self.cfg.sinkhorn_itr)

                else:
                    if self.cfg.use_overlap_as_weights:
                        T = compute_rigid_transform(src_pts, tgt_pts, weights=overlap_prob.squeeze())
                    else:
                        T = compute_rigid_transform(src_pts, tgt_pts, weights=val.permute(1,0).squeeze())
                        # self.weight_matrix += list(val.permute(1,0).squeeze().cpu().numpy())
                        # T = compute_rigid_transform(src_pts, tgt_pts)

                if self.cfg.use_lgr:
                    T = self.local_global_registration(src_pts, tgt_pts, val.permute(1,0).squeeze(), T)

                if self.cfg.use_ransac:
                    T = self.ransac(src_pts, tgt_pts, val.permute(1,0).squeeze())

                overlap_prob_list.append(val.squeeze())
                ind_list.append(ind.squeeze())
                src_pts_list.append(src_pts.squeeze())
                tgt_pts_list.append(tgt_pts.squeeze())

            else:
                if self.dual_normalization:
                    attn_src = torch.nn.functional.softmax(correlation, dim=-2)
                    attn_tgt = torch.nn.functional.softmax(correlation, dim=-1)
                    attn = attn_src * attn_tgt
                else:
                    attn = torch.nn.functional.softmax(correlation, dim=-1)
                
                attn_list.append(attn)

                if self.cfg.use_ratio_test:
                    ind, val = self.ratio_test(attn, dim=2)
                else:
                    val, ind = torch.max(attn, dim=2)
                
                # Threshold the coorelation values?
                if self.cfg.threshold_corr:
                    # val = torch.where(val>self.cfg.corr_threshold, val, 0)
                    val = torch.where(val>torch.median(val), val, 0)

                src_pts = src_xyz[i]
                if self.cfg.use_sinkhorn:
                    tgt_pts = tgt_xyz[i]
                else:
                    try:
                        tgt_pts = torch.gather(tgt_xyz[i], 0, ind.permute(1,0).expand(-1,3))  # [M, 3] -> [N, 3]
                    except:
                        print(f"tgt_xyz shape is: {tgt_xyz.shape}")
                        print(f"ind shape is: {ind.shape}")
                        raise ValueError

                if self.cfg.remove_outliers_overlap:
                    src_overlap_prob = src_overlap_list[i].squeeze(2)
                    tgt_overlap_prob = tgt_overlap_list[i].squeeze(2)
                   
                    tgt_overlap_prob = torch.gather(tgt_overlap_prob, 1, ind)
                    overlap_prob = src_overlap_prob * tgt_overlap_prob

                    if not self.cfg.use_overlap_as_weights:
                        # overlap_mask = overlap_prob > self.cfg.overlap_threshold
                        # val = torch.where(overlap_mask, val, 0)
                        val = val * overlap_prob

                # print(f"src_pts shape is: {src_pts.shape}")
                # print(f"tgt_pts shape is: {tgt_pts.shape}")
                # print(f"new N is: {int(0.2*N)}")
                # print(val.shape)
                if self.cfg.remove_points_from_val:
                    val, ind = torch.topk(val, int(self.cfg.val_threshold*N), dim=1)
                    src_pts = torch.gather(src_pts, 0, ind.permute(1,0).expand(-1,3))
                    tgt_pts = torch.gather(tgt_pts, 0, ind.permute(1,0).expand(-1,3))

                # print(f"src_pts shape is: {src_pts.shape}")
                # print(f"tgt_pts shape is: {tgt_pts.shape}")
                # raise ValueError
                
                # Compute the transformation matrix
                if self.cfg.use_sinkhorn:
                    if self.cfg.use_attn_affinity: # Use attention matrix as the affinity matrix
                        print(src_pts.shape)
                        print(src_xyz[i].shape)
                        print(tgt_pts.shape)
                        raise ValueError
                        affinity = torch.gather(attn, 2, ind.unsqueeze(0).expand(-1,N,-1))
                        T = compute_rigid_transform_with_sinkhorn(src_xyz[i], tgt_pts, affinity, self.cfg.slack, self.cfg.sinkhorn_itr)

                    elif self.cfg.use_corr_affinity: # Compute affinity matrix from the correlation matrix
                        # corr = torch.gather(correlation, 2, ind.unsqueeze(0).expand(-1,N,-1))
                        score_matrix = torch.clamp(1-correlation, min=0.0, max=None)
                        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
                        T = compute_rigid_transform_with_sinkhorn(src_xyz[i], tgt_xyz[i], affinity, self.cfg.slack, self.cfg.sinkhorn_itr)
                    
                    else:
                        # score_matrix = pairwise_distance(src_feats[i], tgt_feats[i])
                        # print(score_matrix.shape)
                        # print(score_matrix.min())
                        # print(score_matrix.max())
                        # raise ValueError
                        score_matrix = torch.matmul(src_feats[i], tgt_feats[i].permute(0, 2, 1)) / (D**0.5)
                        score_matrix = torch.clamp(score_matrix, min=0.0, max=None)

                        # print(score_matrix.min())
                        # print(score_matrix.max())
                        affinity = -(score_matrix - self.softplus(self.alpha))/(torch.exp(self.beta) + 0.02)
                        T = compute_rigid_transform_with_sinkhorn(src_pts.unsqueeze(0), tgt_pts.unsqueeze(0), affinity,self.cfg.slack, self.cfg.sinkhorn_itr)

                else:
                    if self.cfg.use_overlap_as_weights:
                        T = compute_rigid_transform(src_pts, tgt_pts, weights=overlap_prob.squeeze())
                    else:
                        T = compute_rigid_transform(src_pts, tgt_pts, weights=val.permute(1,0).squeeze())
                        # self.weight_matrix += list(val.permute(1,0).squeeze().cpu().numpy())
                        # T = compute_rigid_transform(src_pts, tgt_pts)

                if self.cfg.use_lgr:
                    T = self.local_global_registration(src_pts, tgt_pts, val.permute(1,0).squeeze(), T)

                if self.cfg.use_ransac:
                    T = self.ransac(src_pts, tgt_pts, val.permute(1,0).squeeze())

                overlap_prob_list.append(val.squeeze())
                ind_list.append(ind.squeeze())
                src_pts_list.append(src_pts.squeeze())
                tgt_pts_list.append(tgt_pts.squeeze())

            pose_list.append(T)

        pose_sfc = torch.stack(pose_list, dim=0)
        # print(f"pose_sfc: {pose_sfc.shape}")
        return pose_sfc, attn_list, overlap_prob_list, ind_list, src_pts_list, tgt_pts_list
