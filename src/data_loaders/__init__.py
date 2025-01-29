from .transforms import *
from .modelnet import *
from .collate_functions import collate_pair, collate_tensors
from .threedmatch import ThreeDMatchDataset
from .kitti_pred import KittiDataset
from torch.utils.data.distributed import DistributedSampler

import torchvision


DATASET_CLS = {'3dmatch': ThreeDMatchDataset, 'kitti': KittiDataset}


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
            dataset = modelnet.get_train_datasets(cfg)
        elif phase == 'val':
            dataset = modelnet.get_val_datasets(cfg)
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


def get_multi_dataloader(cfg, phase, num_workers=0, num_gpus=1):
    def get_dataset(_cls, _cfg, _phase):
        if _cls == 'modelnet':
            dataset = getattr(modelnet, f'get_{_phase}_datasets')(_cfg)
        else:
            if _phase == 'train':
                transforms_aug = torchvision.transforms.Compose([
                    transforms.RigidPerturb(perturb_mode=_cfg.perturb_pose),
                    transforms.Jitter(scale=_cfg.augment_noise),
                    transforms.ShufflePoints(),
                    transforms.RandomSwap(),
                ])
            else:
                transforms_aug = None
            dataset = DATASET_CLS[_cls](config=_cfg, phase=_phase, transforms=transforms_aug)

        return dataset

    ds_cls = None
    for key in cfg.datasets.keys():
        ds_cfg = cfg.datasets[key]
        if ds_cls is None:
            ds_cls = get_dataset(key, ds_cfg, phase)
        else:
            ds_cls += get_dataset(key, ds_cfg, phase)

    batch_size = cfg[f'{phase}_batch_size']
    data_loader = torch.utils.data.DataLoader(
        ds_cls,
        batch_size=batch_size,
        shuffle=phase == 'train',
        num_workers=num_workers,
        collate_fn=collate_pair,
        sampler=torch.utils.data.distributed.DistributedSampler(ds_cls) if num_gpus > 1 else None
    )
    return data_loader
