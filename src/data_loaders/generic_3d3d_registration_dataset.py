from typing import Optional, Union
import pickle

from torch.utils.data import Dataset
import numpy as np
from kiss_icp.voxelization import voxel_down_sample

from .array_ops import GridSample, get_3d3d_correspondences_mutual, random_sample_small_transform, \
    get_transform_from_rotation_translation, compose_transforms, apply_transform, min_max_norm, inverse_transform


class Generic3D3DRegistrationDataset(Dataset):
    # Note: if adding an argument to align the point clouds, don't forget to apply same transformation to target
    # coordinates
    def __init__(self,
                 root: str,
                 meta_data: Union[str, None],
                 max_points: Optional[int] = None,
                 max_queries: Optional[int] = None,
                 grid_size: Optional[float] = 0.02,
                 downsample_voxel_size: Optional[float] = None,
                 matching_radius_3d: float = 0.0375,
                 # scene_name: Optional[str] = None,
                 use_augmentation: bool = True,
                 augmentation_noise: float = 0.005,
                 normalize_points: bool = False,
                 bidirectional: bool = False):
        super().__init__()
        self.downsample_voxel_size = downsample_voxel_size
        self.grid_sample = GridSample(grid_size)
        self.root = root
        self.max_points = max_points
        self.max_queries = max_queries
        self.matching_radius_3d = matching_radius_3d
        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.normalize_points = normalize_points
        self.bidirectional = bidirectional

        # Placeholder
        self.meta_data_list = None
        self.data_cache = {}

        if meta_data is not None:
            self.parse_meta_data(meta_data)

    def parse_meta_data(self, filepath) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.meta_data_list = data

    def load_pcd(self, filepath) -> np.ndarray:
        raise NotImplementedError

    def load_pose(self, filepath) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def get_relative_pose(src_pose, tgt_pose):
        src_pose = np.vstack([src_pose, [0., 0., 0., 1.]])
        tgt_pose = np.vstack([tgt_pose, [0., 0., 0., 1.]])

        relative_trans = np.linalg.inv(src_pose) @ tgt_pose
        return relative_trans

    def _trim_num_queries(self, src_queries, tgt_targets):
        assert src_queries.shape[0] == tgt_targets.shape[0]
        if self.max_queries is None:
            return src_queries, tgt_targets

        length = src_queries.shape[0]
        if self.max_queries <= length:
            selected = np.random.choice(length, self.max_queries)
            return src_queries[selected], tgt_targets[selected]
        else:
            selected = np.random.choice(length, self.max_queries - length)
            return np.concatenate([src_queries, src_queries[selected]], axis=0), np.concatenate(
                [tgt_targets, tgt_targets[selected]], axis=0)

    def __len__(self):
        return len(self.meta_data_list)

    def _apply_small_augmentation(self, pcd, scale=0.3):
        aug_transform = random_sample_small_transform(scale=scale)
        pcd_center = pcd.mean(axis=0)
        centralize = get_transform_from_rotation_translation(None, -pcd_center)
        decentralize = get_transform_from_rotation_translation(None, pcd_center)
        aug_transform = compose_transforms(centralize, aug_transform, decentralize)
        pcd = apply_transform(pcd, aug_transform)
        pcd += (np.random.rand(pcd.shape[0], 3) - 0.5) * self.aug_noise

        return pcd, aug_transform

    def construct_data_dict(self, src_pcd, tgt_pcd, tgt2src_transform):
        if self.downsample_voxel_size is not None:
            src_pcd = voxel_down_sample(src_pcd, self.downsample_voxel_size)
            tgt_pcd = voxel_down_sample(tgt_pcd, self.downsample_voxel_size)

        queries, targets = get_3d3d_correspondences_mutual(src_pcd, tgt_pcd, tgt2src_transform,
                                                           self.matching_radius_3d)
        queries, targets = self._trim_num_queries(queries, targets)

        if self.max_points is not None:
            if src_pcd.shape[0] > self.max_points:
                selected = np.random.choice(src_pcd.shape[0], self.max_points)
                src_pcd = src_pcd[selected]

            if tgt_pcd.shape[0] > self.max_points:
                selected = np.random.choice(tgt_pcd.shape[0], self.max_points)
                tgt_pcd = tgt_pcd[selected]

        if self.use_augmentation:
            src_pcd, src_aug = self._apply_small_augmentation(src_pcd)
            queries = apply_transform(queries, src_aug)
            tgt_pcd, tgt_aug = self._apply_small_augmentation(tgt_pcd)
            targets = apply_transform(targets, tgt_aug)
            tgt2src_transform = compose_transforms(inverse_transform(tgt_aug), tgt2src_transform, src_aug)

        if self.normalize_points:
            src_pcd = min_max_norm(src_pcd)
            tgt_pcd = min_max_norm(tgt_pcd)
        src_grid_sample = self.grid_sample(src_pcd)
        tgt_grid_sample = self.grid_sample(tgt_pcd)

        queries = queries.astype(np.float32)
        targets = targets.astype(np.float32)
        norm_queries = min_max_norm(queries).astype(np.float32)
        norm_targets = min_max_norm(targets).astype(np.float32)
        data_dict = {'src_pcd': src_pcd.astype(np.float32),
                     'tgt_pcd': tgt_pcd.astype(np.float32),
                     'tgt2src_transform': tgt2src_transform,
                     'queries': queries,
                     'norm_queries': norm_queries,
                     'targets': targets,
                     'norm_targets': norm_targets,
                     'src_grid_coord': src_grid_sample['grid_coord'],
                     'min_src_grid_coord': src_grid_sample['min_coord'],
                     'tgt_grid_coord': tgt_grid_sample['grid_coord'],
                     'min_tgt_grid_coord': tgt_grid_sample['min_coord']}

        if self.bidirectional:
            data_dict['queries'] = np.concatenate([queries, targets], axis=0)
            data_dict['targets'] = np.concatenate([targets, queries], axis=0)
            data_dict['norm_queries'] = np.concatenate([norm_queries, norm_targets], axis=0)
            data_dict['norm_targets'] = np.concatenate([norm_targets, norm_queries], axis=0)

        return data_dict

    def __getitem__(self, index: int):
        raise NotImplementedError
