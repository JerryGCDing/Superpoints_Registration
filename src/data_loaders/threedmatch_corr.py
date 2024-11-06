import os
from typing import Optional, Union

import numpy as np
import torch

from .generic_3d3d_registration_dataset import Generic3D3DRegistrationDataset


class ThreeDMatchDataset(Generic3D3DRegistrationDataset):
    def __init__(self,
                 root: str,
                 meta_data: Union[str, None],
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 downsample_voxel_size: Optional[float] = None,
                 matching_radius_3d: Optional[float] = 1,
                 use_augmentation: bool = True,
                 normalize_points: bool = False):
        super().__init__(root,
                         meta_data,
                         max_points,
                         max_queries,
                         grid_size,
                         downsample_voxel_size,
                         matching_radius_3d,
                         use_augmentation,
                         normalize_points)

    def load_pcd(self, filepath) -> np.ndarray:
        pcd = torch.load(filepath)
        return pcd

    def load_pose(self, filepath) -> np.ndarray:
        with open(filepath, 'r') as f:
            data = [line.rstrip() for line in f.readlines()[1:]]
        pose = np.fromstring(' '.join(data), sep=' ').reshape(4, 4)
        return pose[:3, :]

    def __len__(self):
        return len(self.meta_data_list['src'])

    def __getitem__(self, index):
        src_pcd = self.load_pcd(os.path.join(self.root, self.meta_data_list['src'][index]))
        tgt_pcd = self.load_pcd(os.path.join(self.root, self.meta_data_list['tgt'][index]))
        src_info = self.meta_data_list['src'][index].split('.')[0] + '.info.txt'
        src_pose = self.load_pose(os.path.join(self.root, src_info))
        tgt_info = self.meta_data_list['tgt'][index].split('.')[0] + '.info.txt'
        tgt_pose = self.load_pose(os.path.join(self.root, tgt_info))
        tgt2src_transform = self.get_relative_pose(src_pose, tgt_pose)

        data_dict = self.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
        data_dict['src'] = self.meta_data_list['src'][index]
        data_dict['tgt'] = self.meta_data_list['tgt'][index]

        return data_dict


if __name__ == '__main__':
    import pickle
    from array_ops import apply_transform
    from utils.visualization import draw_straight_correspondences

    with open('../datasets/3dmatch/train_info.pkl', 'rb') as f:
        data = pickle.load(f)
    threedmatch_demo = ThreeDMatchDataset('./sample_data/3dmatch/', None, max_points=1000, max_queries=None,
                                          use_augmentation=True)
    src_pcd = threedmatch_demo.load_pcd('./sample_data/3dmatch/cloud_bin_0.pth')
    src_pose = threedmatch_demo.load_pose('./sample_data/3dmatch/cloud_bin_0.info.txt')
    tgt_pcd = threedmatch_demo.load_pcd('./sample_data/3dmatch/cloud_bin_1.pth')
    tgt_pose = threedmatch_demo.load_pose('./sample_data/3dmatch/cloud_bin_1.info.txt')

    tgt2src_transform = threedmatch_demo.get_relative_pose(src_pose, tgt_pose)
    data_dict = threedmatch_demo.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)

    draw_straight_correspondences(data_dict['src_pcd'], data_dict['tgt_pcd'], data_dict['queries'],
                                  data_dict['targets'], offsets=(0., 2., 0.))
