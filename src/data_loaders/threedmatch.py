"""Dataloader for 3DMatch dataset

Modified from Predator source code by Shengyu Huang:
  https://github.com/overlappredator/OverlapPredator/blob/main/datasets/indoor.py
"""
import logging
import os
import pickle

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.se3_numpy import se3_init, se3_transform, se3_inv
from src.utils.pointcloud import compute_overlap
from .base.easy_dataset import EasyDataset


class ThreeDMatchDataset(Dataset, EasyDataset):

    def __init__(self, cfg, phase, transforms=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)

        assert phase in ['train', 'val', 'test']
        if phase in ['train', 'val']:
            info_fname = f'src/datasets/3dmatch/{phase}_info.pkl'
            pairs_fname = f'{phase}_pairs-overlapmask.h5'
        else:
            info_fname = f'src/datasets/3dmatch/{phase}_{cfg.benchmark}_info.pkl'
            pairs_fname = f'{phase}_{cfg.benchmark}_pairs-overlapmask.h5'

        with open(info_fname, 'rb') as fid:
            self.infos = pickle.load(fid)

        self.base_dir = None
        if isinstance(cfg.root, str):
            if os.path.exists(f'{cfg.root}/train'):
                self.base_dir = cfg.root
        else:
            for r in cfg.root:
                if os.path.exists(f'{r}/train'):
                    self.base_dir = r
                break
        if self.base_dir is None:
            raise AssertionError(f'Dataset not found in {cfg.root}')
        else:
            self.logger.info(f'Loading data from {self.base_dir}')

        self.cfg = cfg

        self.pairs_data_path = os.path.join(self.base_dir, pairs_fname) if os.path.exists(
            os.path.join(self.base_dir, pairs_fname)) else None
        self.pairs_data = None

        self.search_voxel_size = cfg.overlap_radius
        self.transforms = transforms
        self.phase = phase

    def _load_pairs_data(self):
        self.pairs_data = h5py.File(self.pairs_data_path, 'r')

    def __len__(self):
        return len(self.infos['rot'])

    def __getitem__(self, item):
        # get transformation and point cloud
        pose = se3_init(self.infos['rot'][item], self.infos['trans'][item])  # transforms src to tgt
        pose_inv = se3_inv(pose)
        src_path = self.infos['src'][item]
        tgt_path = self.infos['tgt'][item]
        src_xyz = torch.load(os.path.join(self.base_dir, src_path), map_location='cpu')
        tgt_xyz = torch.load(os.path.join(self.base_dir, tgt_path), map_location='cpu')
        overlap_p = self.infos['overlap'][item]

        if self.pairs_data is None and self.pairs_data_path is not None:
            self._load_pairs_data()

        # Get overlap region
        if self.pairs_data is None:
            src_overlap_mask, tgt_overlap_mask, src_tgt_corr = compute_overlap(
                se3_transform(pose, src_xyz),
                tgt_xyz,
                self.search_voxel_size,
            )
        else:
            src_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/src_mask'])
            tgt_overlap_mask = np.asarray(self.pairs_data[f'pair_{item:06d}/tgt_mask'])
            src_tgt_corr = np.asarray(self.pairs_data[f'pair_{item:06d}/src_tgt_corr'])
        # print("np array: ", src_overlap_mask.shape, tgt_overlap_mask.shape)

        data_pair = {
            'src_xyz': torch.from_numpy(src_xyz).float(),
            'tgt_xyz': torch.from_numpy(tgt_xyz).float(),
            'src_overlap': torch.from_numpy(src_overlap_mask),
            'tgt_overlap': torch.from_numpy(tgt_overlap_mask),
            'correspondences': torch.from_numpy(src_tgt_corr),  # indices
            'pose': torch.from_numpy(pose).float(),
            'idx': item,
            # 'src_path': src_path,  # Uncomment for test
            # 'tgt_path': tgt_path,  # Uncomment for test
            # 'overlap_p': overlap_p,
        }
        # print("torch tensor: ", data_pair['src_overlap'].shape, data_pair['tgt_overlap'].shape)
        if self.transforms is not None:
            self.transforms(data_pair)  # Apply data augmentation

        return data_pair

    def __del__(self):
        """Close file handle when the dataset is deleted."""
        if self.pairs_data is not None:
            self.pairs_data.close()
