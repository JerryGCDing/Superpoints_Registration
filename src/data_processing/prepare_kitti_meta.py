import os
from tqdm import tqdm

import numpy as np

from utils import dump_pickle

def prepare_kitti_meta(kitti_dir, min_displacement=10, max_displacement=20):
    kitti_dir = os.path.join(kitti_dir, 'dataset')
    seq_dir = os.path.join(kitti_dir, 'sequences')
    pose_dir = os.path.join(kitti_dir, 'pose')
    seq_id = sorted(os.listdir(seq_dir))

    sample_meta_list = []
    for seq in tqdm(seq_id, 'Generating point cloud pairs'):
        pose_file = os.path.join(pose_dir, f'{seq:02d}.txt')
        all_poses = np.genfromtxt(pose_file).astype(np.float32).reshape(-1, 3, 4)
        # N, 3
        all_transitions = np.squeeze(all_poses[:, :, -1])
        co_transitions = all_transitions[None, :, :] - all_transitions[:, None, :]
        co_distances = np.linalg.norm(co_transitions, axis=-1)
        upper_tri_indices = np.triu_indices_from(co_distances)

        co_distance_mask = (co_distances >= min_displacement) & (co_distances <= max_displacement)
        selected_pairs = [(i, j) for i, j in zip(*upper_tri_indices) if co_distance_mask[i, j]]

        for src, tgt in selected_pairs:
            sample_meta_list.append({'seq': seq, 'src_frame': src, 'tgt_frame': tgt})

        print(f'{len(selected_pairs)} pairs generated for sequence {seq}.')

    dump_pickle(sample_meta_list, './kitti_meta.pkl')
