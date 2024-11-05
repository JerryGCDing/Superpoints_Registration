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
                 downsample_voxel_size: Optional[float] = 0.2,
                 matching_radius_3d: Optional[float] = None,
                 use_augmentation: bool = True,
                 normalize_points: bool = False,
                 remove_ground: bool = False):
        super().__init__(root,
                         meta_data,
                         max_points,
                         max_queries,
                         downsample_voxel_size,
                         grid_size,
                         matching_radius_3d,
                         use_augmentation,
                         normalize_points)
        assert split in ('train', 'val', 'test')
        self.root = os.path.join(self.root, 'dataset')
        self.remove_ground = remove_ground
        # seq_id, src_frame_id, tgt_frame_id
        self.meta_data_list = self.meta_data_list[self.DATA_SPLITS[split]] if meta_data is not None else None

    def load_pcd(self, filepath) -> np.ndarray:
        data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)[:, :3]
        if self.remove_ground:
            data = data[data[:, -1] > -1]

        return data

    @staticmethod
    def load_velo2cam(filepath):
        with open(filepath, 'r') as f:
            Tr = f.readlines()[-1].strip()
        assert Tr.startswith('Tr:')
        velo2cam = np.fromstring(Tr[4:], sep=' ').reshape(3, 4)

        return velo2cam

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
        velo2cam = self.load_velo2cam(os.path.join(seq_dir, 'calib.txt'))
        tgt2src_transform = self.get_relative_pose(src_pose, tgt_pose)

        src_pcd = self.load_pcd(os.path.join(seq_dir, 'velodyne', f'{src_frame_idx:06d}.bin'))
        tgt_pcd = self.load_pcd(os.path.join(seq_dir, 'velodyne', f'{tgt_frame_idx:06d}.bin'))

        data_dict = self.construct_data_dict(apply_transform(src_pcd, velo2cam), apply_transform(tgt_pcd, velo2cam),
                                             tgt2src_transform)
        data_dict['seq'] = sample_meta['seq']
        data_dict['src_frame'] = src_frame_idx
        data_dict['tgt_frame'] = tgt_frame_idx

        return data_dict


if __name__ == '__main__':
    from array_ops import apply_transform
    from utils.visualization import draw_correspondences

    kitti_demo = KittiDataset('./sample_data/kitti/', None, 'train', max_points=1000, max_queries=100,
                              use_augmentation=True, remove_ground=True)
    src_pcd_idx = 0
    src_pcd = kitti_demo.load_pcd(f'./sample_data/kitti/sequences/velodyne/{src_pcd_idx:06d}.bin')
    tgt_pcd_idx = 50
    tgt_pcd = kitti_demo.load_pcd(f'./sample_data/kitti/sequences/velodyne/{tgt_pcd_idx:06d}.bin')

    src_pose = kitti_demo.load_pose('./sample_data/kitti/poses/00.txt', src_pcd_idx)
    tgt_pose = kitti_demo.load_pose('./sample_data/kitti/poses/00.txt', tgt_pcd_idx)
    transition = np.linalg.norm(src_pose[:, -1] - tgt_pose[:, -1], axis=-1)
    print(transition)
    tgt2src_transform = kitti_demo.get_relative_pose(src_pose, tgt_pose)
    velo2cam = kitti_demo.load_velo2cam('./sample_data/kitti/sequences/calib.txt')
    src_pcd = apply_transform(src_pcd, velo2cam)
    tgt_pcd = apply_transform(tgt_pcd, velo2cam)

    data_dict = kitti_demo.construct_data_dict(src_pcd, tgt_pcd, tgt2src_transform)
    # src_pcd_viz = make_open3d_point_cloud(data_dict['src_pcd'])
    # src_pcd_viz.paint_uniform_color(get_color('custom_yellow'))
    # tgt_pcd_align = apply_transform(data_dict['tgt_pcd'], data_dict['tgt2src_transform'])
    # tgt_pcd_viz = make_open3d_point_cloud(tgt_pcd_align)
    # tgt_pcd_viz.paint_uniform_color(get_color('custom_blue'))
    #
    # draw_geometries(src_pcd_viz, tgt_pcd_viz)

    tgt_pcd_align = apply_transform(data_dict['tgt_pcd'], data_dict['tgt2src_transform'])
    draw_correspondences(data_dict['src_pcd'], tgt_pcd_align, data_dict['src_corr_indices'],
                         data_dict['tgt_corr_indices'], offsets=(0., 20., 0.))
