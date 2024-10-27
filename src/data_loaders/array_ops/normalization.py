import numpy as np
from typing import Optional, Union, Tuple, List


def min_max_norm(
        a: np.ndarray,
        *,
        axis: int = 0,
        min_val: Optional[Union[np.ndarray, Tuple, List]] = None,
        max_val: Optional[Union[np.ndarray, Tuple, List]] = None
):
    a = a.astype(np.float32)
    min_val = np.min(a, axis=axis, keepdims=True) if min_val is None else np.asarray(min_val)
    max_val = np.max(a, axis=axis, keepdims=True) if max_val is None else np.asarray(max_val)

    norm = (a - min_val) / (max_val - min_val)
    return norm

def center_shift(points, apply_z=True):
    x_min, y_min, z_min = points.min(axis=0)
    x_max, y_max, _ = points.max(axis=0)
    
    if apply_z:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
    else:
        shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]

    points -= shift
    return points    