from typing import Optional, Union

import numpy as np
from numpy import ndarray

from .knn import knn
from .se3 import apply_transform


def registration_corr_distance(src_corr_points, tgt_corr_points, transform):
    """Computing the mean distance between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    distances = np.sqrt(((tgt_corr_points - src_corr_points) ** 2).sum(1))
    mean_distance = np.mean(distances)
    return mean_distance


def registration_inlier_ratio(src_corr_points, tgt_corr_points, transform, positive_radius=0.1):
    """Computing the inlier ratio between a set of correspondences."""
    src_corr_points = apply_transform(src_corr_points, transform)
    corr_distances = np.sqrt(((tgt_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(corr_distances < positive_radius)
    return inlier_ratio


def point_cloud_overlap(src_points, tgt_points, transform=None, positive_radius=0.1):
    """Compute the overlap of two point clouds."""
    if transform is not None:
        src_points = apply_transform(src_points, transform)
    nn_distances, _ = knn(tgt_points, src_points, k=1, return_distance=True)
    overlap = np.mean(nn_distances < positive_radius)
    return overlap
