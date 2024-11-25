import torch

import data_loaders.transforms
from data_loaders.collate_functions import collate_pair, collate_tensors, \
    PointCloudRegistrationCollateFn
from .modelnet_corr import ModelNetDataset
from .threedmatch_corr import ThreeDMatchDataset
from .kitti_corr import KITTIDataset
from torch.utils.data.distributed import DistributedSampler

import torchvision


def get_dataloader(cfg, stage, num_workers=0, num_gpus=1):
    assert stage in ('train', 'val')
    if cfg.dataset == '3dmatch':
        dataset = ThreeDMatchDataset(root=cfg.root,
                                     meta_data=cfg[f'{stage}_meta_data'],
                                     max_points=cfg.max_points,
                                     max_queries=cfg.max_queries,
                                     grid_size=cfg.grid_size,
                                     downsample_voxel_size=cfg.downsample_voxel_size,
                                     matching_radius_3d=cfg.matching_radius_3d,
                                     use_augmentation=cfg.use_augmentation,
                                     normalize_points=cfg.normalize_points,
                                     bidirectional=cfg.bidirectional)

    elif cfg.dataset == 'modelnet':
        dataset = ModelNetDataset(root=cfg.root,
                                  meta_data=cfg[f'{stage}_meta_data'],
                                  max_points=cfg.max_points,
                                  max_queries=cfg.max_queries,
                                  grid_size=cfg.grid_size,
                                  downsample_voxel_size=cfg.downsample_voxel_size,
                                  matching_radius_3d=cfg.matching_radius_3d,
                                  use_augmentation=cfg.use_augmentation,
                                  normalize_points=cfg.normalize_points,
                                  bidirectional=cfg.bidirectional)

    elif cfg.dataset == "kitti":
        dataset = KITTIDataset(root=cfg.root,
                               meta_data=cfg.meta_data,
                               split=cfg.split,
                               max_points=cfg.max_points,
                               max_queries=cfg.max_queries,
                               grid_size=cfg.grid_size,
                               downsample_voxel_size=cfg.downsample_voxel_size,
                               matching_radius_3d=cfg.matching_radius_3d,
                               use_augmentation=cfg.use_augmentation,
                               normalize_points=cfg.normalize_points,
                               bidirectional=cfg.bidirectional,
                               remove_ground=cfg.remove_ground)

    else:
        raise NotImplementedError

    # # For calibrating the number of neighbors (set in config file)
    # from models.backbone_kpconv.kpconv import calibrate_neighbors
    # neighborhood_limits = calibrate_neighbors(dataset, cfg)
    # print(f"Neighborhood limits: {neighborhood_limits}")
    # raise ValueError

    batch_size = cfg[f'{stage}_batch_size']
    shuffle = stage == 'train'

    collate_fn = PointCloudRegistrationCollateFn(
        ('tgt2src_transform', 'queries', 'norm_queries', 'targets', 'norm_targets'))
    if cfg.model in ["regtr.RegTR", "qk_regtr.RegTR", "qk_regtr_old.RegTR", "qk_regtr_overlap.RegTR",
                     "qk_regtr_full.RegTR", "qk_regtr_full_pointformer.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if num_gpus > 1 else None
        )
    elif cfg.model in ["qk_revvit.RegTR", "qk_revvit_2.RegTR", "qk_ce.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_tensors,
            sampler=torch.utils.data.distributed.DistributedSampler(dataset) if num_gpus > 1 else None
        )
    else:
        raise NotImplementedError

    return data_loader
