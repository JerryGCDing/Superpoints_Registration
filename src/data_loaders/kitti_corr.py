import os
from typing import Optional, Union

import numpy as np

from .generic_3d3d_registration_dataset import Generic3D3DRegistrationDataset


class KittiDataset(Generic3D3DRegistrationDataset):
    DATA_SPLITS = {
        'train': slice(0, 8),
        'val': slice(8, 11)
    }

    def __init__(self,
                 root: str,
                 meta_data: Union[str, None],
                 split: str,
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 matching_radius_3d: float = 0.0375,
                 use_augmentation: bool = True,
                 normalize_points: bool = False):
        super().__init__(root,
                         meta_data,
                         max_points,
                         max_queries,
                         grid_size,
                         matching_radius_3d,
                         use_augmentation,
                         normalize_points)
        assert split in ('train', 'val', 'test')
        self.root = os.path.join(self.root, 'dataset')
        # seq_id, src_frame_id, tgt_frame_id
        self.meta_data_list = self.meta_data_list[self.DATA_SPLITS[split]]

    def load_pcd(self, filepath) -> np.ndarray:
        data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
        return data[:, :3]

    def load_pose(self, filepath, index=None) -> np.ndarray:
        if filepath not in self.data_cache:
            self.data_cache[filepath] = np.genfromtxt(filepath).astype(np.float32).reshape(-1, 3, 4)

        if index is not None:
            return self.data_cache[filepath][index]
        else:
            return self.data_cache[filepath]

    def __getitem__(self, index):
        sample_meta = self.meta_data_list[index]
        seq_dir = os.path.join(self.root, 'sequences', f"{sample_meta['seq']}")
        pose_file = os.path.join(self.root, 'poses', f"{sample_meta['seq']}.txt")

        src_frame_idx = sample_meta['src_frame']
        tgt_frame_idx = sample_meta['tgt_frame']

        src_pose = self.load_pose(pose_file, src_frame_idx)
        tgt_pose = self.load_pose(pose_file, tgt_frame_idx)
        tgt2src_transform = self.get_relative_pose(src_pose, tgt_pose)

        src_pcd = self.load_pcd(os.path.join(seq_dir, 'velodyne', f'{src_frame_idx:06d}.bin'))
        tgt_pcd = self.load_pcd(os.path.join(seq_dir, 'velodyne', f'{tgt_frame_idx:06d}.bin'))

        data_dict = self.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
        data_dict['seq'] = sample_meta['seq']
        data_dict['src_frame'] = src_frame_idx
        data_dict['tgt_frame'] = tgt_frame_idx

        return data_dict


if __name__ == '__main__':
    kitti_demo = KittiDataset('./sample_data/kitti/', None, 'train', max_queries=100, use_augmentation=False)
