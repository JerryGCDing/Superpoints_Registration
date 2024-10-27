import numpy as np


class GridSample(object):
    def __init__(
        self,
        grid_size=0.05,
    ):
        self.grid_size = grid_size

    def __call__(self, pcd):
        sign_mask = np.sign(pcd)
        abs_pcd = abs(pcd)
        abs_scaled_coord = abs_pcd / np.array(self.grid_size)
        abs_grid_coord = np.floor(abs_scaled_coord).astype(int)
        grid_coord = sign_mask * abs_grid_coord
        min_coord = grid_coord.min(0)
        grid_coord -= min_coord
        min_coord = min_coord * np.array(self.grid_size)

        ret_dict = {'grid_coord': grid_coord, 'min_coord': min_coord.reshape([1, 3])}
        return ret_dict
