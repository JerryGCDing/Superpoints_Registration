import os
from typing import Optional, Union

import numpy as np

import h5py

from .generic_3d3d_registration_dataset import Generic3D3DRegistrationDataset
from .array_ops import inverse_transform, min_max_norm


class ModelNetDataset(Generic3D3DRegistrationDataset):
    def __init__(self,
                 root: str,
                 meta_data: Union[str, None],
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 downsample_voxel_size: Optional[float] = None,
                 matching_radius_3d: Optional[float] = 0.0375,
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

    def parse_meta_data(self, filepath) -> None:
        with open(filepath, 'r') as f:
            files = f.readlines()

        pcd_data = []
        for file in files:
            pcd_data.append(self.load_pcd(os.path.join(self.root, file)))
        self.data_cache['pcd'] = [sample for sample in np.concatenate(pcd_data, axis=0)]

    def load_pcd(self, filepath) -> np.ndarray:
        with h5py.File(filepath, 'r') as h5:
            pcd = np.asarray(h5['data'], dtype=np.float32)
        return pcd

    def __len__(self):
        return len(self.data_cache['pcd'])

    def __getitem__(self, index):
        src_pcd = self.data_cache['pcd'][index]
        tgt_pcd, transform = self._apply_small_augmentation(src_pcd.copy(), scale=0.5)
        tgt2src_transform = inverse_transform(transform)
        queries = src_pcd.copy()
        targets = tgt_pcd.copy()

        if self.max_points is not None and src_pcd.shape[0] > self.max_points:
            src_selected = np.zeros(src_pcd.shape[0], dtype=bool)
            src_selected[np.random.choice(src_pcd.shape[0], self.max_points)] = 1
            tgt_selected = np.zeros(tgt_pcd.shape[0], dtype=bool)
            tgt_selected[np.random.choice(tgt_pcd.shape[0], self.max_points)] = 1
            mutual_selected = src_selected & tgt_selected

            src_pcd = src_pcd[src_selected]
            tgt_pcd = tgt_pcd[tgt_selected]
            queries = queries[mutual_selected]
            targets = targets[mutual_selected]

        queries, targets = self._trim_num_queries(queries, targets)

        if self.normalize_points:
            src_pcd = min_max_norm(src_pcd)
            tgt_pcd = min_max_norm(tgt_pcd)
        src_grid_sample = self.grid_sample(src_pcd)
        tgt_grid_sample = self.grid_sample(tgt_pcd)

        data_dict = {'src_pcd': src_pcd,
                     'tgt_pcd': tgt_pcd,
                     'tgt2src_transform': tgt2src_transform,
                     'queries': queries,
                     'norm_queries': min_max_norm(queries),
                     'targets': targets,
                     'norm_targets': min_max_norm(targets),
                     'src_grid_coord': src_grid_sample['grid_coord'],
                     'min_src_grid_coord': src_grid_sample['min_coord'],
                     'tgt_grid_coord': tgt_grid_sample['grid_coord'],
                     'min_tgt_grid_coord': tgt_grid_sample['min_coord']}

        return data_dict


if __name__ == '__main__':
    from array_ops import apply_transform
    from utils.visualization import draw_straight_correspondences

    modelnet_demo = ModelNetDataset('./sample_data/modelnet/', None, max_points=1000, max_queries=None)
    src_pcd = modelnet_demo.load_pcd(os.path.join(modelnet_demo.root, 'ply_data_train0.h5'))[0]
    tgt_pcd, transform = modelnet_demo._apply_small_augmentation(src_pcd.copy(), scale=0.5)
    tgt2src_transform = inverse_transform(transform)
    queries = src_pcd.copy()
    targets = tgt_pcd.copy()

    if modelnet_demo.max_points is not None and src_pcd.shape[0] > modelnet_demo.max_points:
        src_selected = np.zeros(src_pcd.shape[0], dtype=bool)
        src_selected[np.random.choice(src_pcd.shape[0], modelnet_demo.max_points)] = 1
        tgt_selected = np.zeros(tgt_pcd.shape[0], dtype=bool)
        tgt_selected[np.random.choice(tgt_pcd.shape[0], modelnet_demo.max_points)] = 1
        mutual_selected = src_selected & tgt_selected

        src_pcd = src_pcd[src_selected]
        tgt_pcd = tgt_pcd[tgt_selected]
        queries = queries[mutual_selected]
        targets = targets[mutual_selected]

    queries, targets = modelnet_demo._trim_num_queries(queries, targets)
    tgt_pcd = apply_transform(tgt_pcd, tgt2src_transform)
    targets = apply_transform(targets, tgt2src_transform)
    draw_straight_correspondences(src_pcd, tgt_pcd, queries, targets, offsets=(0., 2., 0.))
