import torch
from torch.utils.data.dataloader import default_collate
from typing import Sequence, Mapping
# import MinkowskiEngine as ME


def collate_fn(batch):
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
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch = {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
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
