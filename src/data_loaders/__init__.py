from .transforms import *
from .modelnet import *
from .collate_functions import collate_pair, collate_tensors
from .threedmatch import ThreeDMatchDataset
from .kitti_pred import KittiDataset
from torch.utils.data.distributed import DistributedSampler

import torchvision


def get_dataloader(cfg, phase, num_workers=0, num_gpus=1):
    assert phase in ['train', 'val', 'test']

    if cfg.dataset == '3dmatch':
        if phase == 'train':
            # Apply training data augmentation (Pose perturbation and jittering)
            transforms_aug = torchvision.transforms.Compose([
                transforms.RigidPerturb(perturb_mode=cfg.perturb_pose),
                transforms.Jitter(scale=cfg.augment_noise),
                transforms.ShufflePoints(),
                transforms.RandomSwap(),
            ])
        else:
            transforms_aug = None

        dataset = ThreeDMatchDataset(
            cfg=cfg,
            phase=phase,
            transforms=transforms_aug,
        )

    elif cfg.dataset == 'modelnet':
        if phase == 'train':
            dataset = modelnet.get_train_datasets(cfg)[0]
        elif phase == 'val':
            dataset = modelnet.get_train_datasets(cfg)[1]
        elif phase == 'test':
            dataset = modelnet.get_test_datasets(cfg)

    elif cfg.dataset == "kitti":
        if phase == 'train':
            # Apply training data augmentation (Pose perturbation and jittering)
            transforms_aug = torchvision.transforms.Compose([
                transforms.RigidPerturb(perturb_mode=cfg.perturb_pose),
                transforms.Jitter(scale=cfg.augment_noise),
                transforms.ShufflePoints(),
                transforms.RandomSwap(),
            ])
        else:
            transforms_aug = None
        dataset = KittiDataset(config=cfg, phase=phase, transforms=transforms_aug)

    else:
        raise AssertionError('Invalid dataset')

    # # For calibrating the number of neighbors (set in config file)
    # from models.backbone_kpconv.kpconv import calibrate_neighbors
    # neighborhood_limits = calibrate_neighbors(dataset, cfg)
    # print(f"Neighborhood limits: {neighborhood_limits}")
    # raise ValueError

    batch_size = cfg[f'{phase}_batch_size']
    shuffle = phase == 'train'
    shuffle = False

    if cfg.model in ["regtr.RegTR", "qk_regtr.RegTR", "qk_regtr_old.RegTR", "qk_regtr_overlap.RegTR",
                     "qk_regtr_full.RegTR"]:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if num_gpus == 1 else False,
            num_workers=num_workers,
            collate_fn=collate_pair,
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
