import numpy as np

class InitWSI:
    def __init__(self):
        pass

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
