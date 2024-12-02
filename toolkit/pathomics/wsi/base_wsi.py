from abc import ABC
from pathlib import Path

import numpy as np


class BaseWSI(ABC):
    def __init__(self, wsi_path: Path, tissue_geom=None):
        super().__init__()
        self.tissue_geom = tissue_geom
        self._wsi_path = Path(wsi_path)
        self.name = Path(self._wsi_path.name)
        self.stem = Path(self._wsi_path.stem)

    def get_dims_at_mpp(self, target_mpp):
        scale, rescale = self.scale_mpp(target_mpp)
        scaled_dims = self.get_dims_at_scale(scale)
        return scaled_dims

    def get_dims_at_scale(self, scale):
        return (
            int((np.array(self.dims) * scale)[0]),
            int((np.array(self.dims) * scale)[1]),
        )

    def factor_mpp(self, target_mpp, source_mpp=None):
        if source_mpp is None:
            factor = target_mpp / self.mpp
        else:
            factor = target_mpp / source_mpp
        return factor

    def scale_mpp(self, target_mpp):
        rescale = self.factor_mpp(target_mpp)
        scale = 1 / rescale
        return scale, rescale
        
    @staticmethod
    def round_to_nearest_even(x):
        return round(x / 2) * 2
