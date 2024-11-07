from misc import round_to_nearest_even

class Slicer:
    def __init__(self, wsi):
        super(Slicer, self).__init__()
        self.wsi = wsi

    def get_slicing_params_single(self, target_mpp, patch_size, overlap_size, context_size):
        
        self._target_mpp = target_mpp
        self._patch_dims = (patch_size, patch_size)
        self._overlap_dims = (overlap_size, overlap_size)
        self._context_dims = (context_size, context_size)
        
        return self._get_slicing_params()

    def get_slicing_params_tuples(
        self, target_mpp: tuple, patch_dims: tuple, overlap_dims: tuple, context_dims: tuple
    ):
        self._target_mpp = target_mpp
        self._patch_dims = patch_dims
        self._overlap_dims = overlap_dims
        self._context_dims = context_dims
        return self._get_slicing_params()
    
    def _get_slicing_params(self):
        """
        factor1 : factor to downsample from original_mpp to target_mpp
        factor2 : factor to downsample from original_mpp to downsample_mpp
        factor3 : factor to downsample from downsample_mpp to target_mpp
        """

        slicing_params = {}

        slicing_params["shift_dims"] = (
            self._context_dims[0] + self._overlap_dims[0],
            self._context_dims[1] + self._overlap_dims[1],
        )

        slicing_params["extraction_dims"] = (
            self._patch_dims[0] + 2 * self._context_dims[0],
            self._patch_dims[1] + 2 * self._context_dims[1],
        )

        slicing_params["stride_dims"] = (
            self._patch_dims[0]
            - (self._overlap_dims[0] + 2 * self._context_dims[0]),
            self._patch_dims[1]
            - (self._overlap_dims[1] + 2 * self._context_dims[1]),
        )

        slicing_params["factor1"] = self.wsi.factor_mpp(self._target_mpp)
        slicing_params["level"] = self.wsi.get_level_for_downsample(slicing_params["factor1"])
        slicing_params["level_dims"] = self.wsi.get_level_dimensions()[slicing_params["level"]]
        slicing_params["mpp_at_level"] = (
            self.wsi.get_level_downsamples()[slicing_params["level"]] * self.wsi.mpp
        )
        slicing_params["factor2"] = self.wsi.factor_mpp(slicing_params["mpp_at_level"])
        slicing_params["factor3"] = self.wsi.factor_mpp(
            target_mpp=self._target_mpp, source_mpp=slicing_params["mpp_at_level"]
        )

        slicing_params["extraction_dims_at_level"] = (
            round_to_nearest_even(slicing_params["extraction_dims"][0] * slicing_params["factor3"]),
            round_to_nearest_even(slicing_params["extraction_dims"][1] * slicing_params["factor3"]),
        )

        slicing_params["stride_dims_at_level"] = (
            round_to_nearest_even(slicing_params["stride_dims"][0] * slicing_params["factor3"]),
            round_to_nearest_even(slicing_params["stride_dims"][1] * slicing_params["factor3"]),
        )

        return slicing_params
