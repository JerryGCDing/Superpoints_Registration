import torch
from torch.utils.data.dataloader import default_collate
from typing import Sequence, Mapping, List, Callable, Optional
# import MinkowskiEngine as ME
import numpy as np
from itertools import chain


def array_to_tensor(x):
    """Convert all numpy arrays to pytorch tensors."""
    if isinstance(x, list):
        x = [array_to_tensor(item) for item in x]
    elif isinstance(x, tuple):
        x = tuple([array_to_tensor(item) for item in x])
    elif isinstance(x, dict):
        x = {key: array_to_tensor(value) for key, value in x.items()}
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    return x

def collate_dict(data_dicts: List[dict]) -> dict:
    """Collate a batch of dict.

    The collated dict contains all keys from the batch, with each key mapped to a list of data. If a certain key is
    missing in one dict, `None` is used for padding so that all lists have the same length (the batch size).

    Args:
        data_dicts (List[dict]): A batch of data dicts.

    Returns:
        A dict with all data collated.
    """
    keys = set(chain(*[list(data_dict.keys()) for data_dict in data_dicts]))
    collated_dict = {key: [data_dict.get(key) for data_dict in data_dicts] for key in keys}
    return collated_dict

def ptv3_collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ptv3_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: ptv3_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch.keys():
            if "offset" in key:
                batch[key] = torch.cumsum(batch[key], dim=0)
        return batch
    else:
        return default_collate(batch)

def collate_pair(list_data):
    """Collates data using a list, for tensors which are of different sizes
    (e.g. different number of points). Otherwise, stacks them as per normal.
    """
    batch_sz = len(list_data)

    # Collate as normal, other than fields that cannot be collated due to differing sizes,
    # we retain it as a python list
    to_retain_as_list = ['src_points', 'tgt_points',
                         'src_grid', 'tgt_grid',
                         'src_length', 'tgt_length',
                         'src_xyz', 'tgt_xyz',
                         'src_overlap', 'tgt_overlap',
                         'tgt_raw',
                         'correspondences',
                         'src_path', 'tgt_path',
                         'idx']

    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    
    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data

def collate_tensors(list_data):
    """
    Collates the modelnet dataset into a stack of tensors since each pointcloud in modelnet is of the same size
    """

    batch_sz = len(list_data)

    to_retain_as_list = []
    data = {k: [list_data[b][k] for b in range(batch_sz)] for k in to_retain_as_list if k in list_data[0]}
    data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
    
    data['src_xyz'] =  torch.stack([list_data[b]['src_xyz'].T for b in range(batch_sz)], dim=0)
    data['tgt_xyz'] =  torch.stack([list_data[b]['tgt_xyz'].T for b in range(batch_sz)], dim=0)
    data['tgt_raw'] =  torch.stack([list_data[b]['tgt_raw'].T for b in range(batch_sz)], dim=0)


    if 'overlap_p' in list_data[0]:
        data['overlap_p'] = torch.tensor([list_data[b]['overlap_p'] for b in range(batch_sz)])
    return data

# def collate_sparse_tensors(list_data):
#     batch_sz = len(list_data)
#     data = {}
#     coords_src = [list_data[b]['coords_src'] for b in range(batch_sz)]
#     feats_src = [list_data[b]['feats_src'] for b in range(batch_sz)]
#     coords_tgt = [list_data[b]['coords_tgt'] for b in range(batch_sz)]
#     feats_tgt = [list_data[b]['feats_tgt'] for b in range(batch_sz)]
#
#     data['coords_src'], data['feats_src'] = ME.utils.sparse_collate(coords=coords_src, feats=feats_src)
#     data['coords_tgt'], data['feats_tgt'] = ME.utils.sparse_collate(coords=coords_tgt, feats=feats_tgt)
#
#     data['pose'] = torch.stack([list_data[b]['pose'] for b in range(batch_sz)], dim=0)  # (B, 3, 4)
#
#     data['src_xyz'] =  torch.stack([list_data[b]['src_xyz'] for b in range(batch_sz)], dim=0)
#     data['tgt_xyz'] =  torch.stack([list_data[b]['tgt_xyz'] for b in range(batch_sz)], dim=0)
#     data['tgt_raw'] =  torch.stack([list_data[b]['tgt_raw'] for b in range(batch_sz)], dim=0)
#
#     return data


class PointCloudRegistrationCollateFn(Callable):
    def __init__(self,
                 batch_keys: Optional[Sequence[str]] = None):
        self.batch_keys = batch_keys

    def __call__(self,
                 data_dicts: List[dict]):
        batch_size = len(data_dicts)

        # 1. collate dict
        collated_dict = collate_dict(data_dicts)

        if batch_size == 1:
            collated_dict = {key: value[0] for key, value in collated_dict.items()}
            collated_dict["src_length"] = np.asarray([collated_dict["src_pcd"].shape[0]])
            collated_dict["tgt_length"] = np.asarray([collated_dict["tgt_pcd"].shape[0]])
        else:
            src_points_list = collated_dict.pop("src_pcd")
            tgt_points_list = collated_dict.pop("tgt_pcd")
            collated_dict["src_pcd"] = np.concatenate(src_points_list, axis=0)
            collated_dict["tgt_pcd"] = np.concatenate(tgt_points_list, axis=0)
            collated_dict["src_length"] = np.asarray([points.shape[0] for points in src_points_list])
            collated_dict["tgt_length"] = np.asarray([points.shape[0] for points in tgt_points_list])

            # additional attributes
            collated_dict["src_grid_coord"] = np.concatenate(collated_dict.pop("src_grid_coord"), axis=0)
            collated_dict["tgt_grid_coord"] = np.concatenate(collated_dict.pop("tgt_grid_coord"), axis=0)

        collated_dict["batch_size"] = batch_size
        if self.batch_keys is not None:
            for key in self.batch_keys:
                if batch_size > 1:
                    collated_dict[key] = np.stack(collated_dict.pop(key), axis=0)
                else:
                    collated_dict[key] = collated_dict.pop(key)[None]

        # 4. array to tensor
        collated_dict = array_to_tensor(collated_dict)

        return collated_dict
