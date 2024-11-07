from cv2 import resize
from numpy import array
from misc import round_to_nearest_even

class InitSlicer():
    def __init__(self, wsi):
        
        self.wsi = wsi
        self.sph = {}  # Slice Parameters History
        self.default_slice_key = -1
        self.set_params_init = False
        self.recent_slice_key = None

    def set_params(
        self,
        target_mpp,
        patch_size,
        overlap_size,
        context_size,
        slice_key=None,
        input_tuple=False,
    ):
        self.set_params_init = True
        self.default_slice_key += 1
        if slice_key is None:
            slice_key = self.default_slice_key

        self.recent_slice_key = slice_key
        
        self.sph[slice_key] = {}
        self.sph[slice_key]["params"] = {}

        self.sph[slice_key]["params"]["target_mpp"] = target_mpp

        if input_tuple:
            self.sph[slice_key]["params"]["patch_dims"] = patch_dims
            self.sph[slice_key]["params"]["overlap_dims"] = overlap_dims
            self.sph[slice_key]["params"]["context_dims"] = context_dims
        else:
            self.sph[slice_key]["params"]["patch_dims"] = (patch_size, patch_size)
            self.sph[slice_key]["params"]["overlap_dims"] = (
                overlap_size,
                overlap_size,
            )
            self.sph[slice_key]["params"]["context_dims"] = (
                context_size,
                context_size,
            )

        self._set_params(slice_key=slice_key)
        self.sph[slice_key]["coordinates"] = self.wsi._get_slice_wsi_coordinates(
            self.sph[slice_key]["params"]
        )

    def _set_params(self, slice_key):
        """
        factor1 : factor to downsample from original_mpp to target_mpp
        factor2 : factor to downsample from original_mpp to downsample_mpp
        factor3 : factor to downsample from downsample_mpp to target_mpp
        """

        self.sph[slice_key]["params"]["shift_dims"] = (
            self.sph[slice_key]["params"]["context_dims"][0]
            + self.sph[slice_key]["params"]["overlap_dims"][0],
            self.sph[slice_key]["params"]["context_dims"][1]
            + self.sph[slice_key]["params"]["overlap_dims"][1],
        )

        self.sph[slice_key]["params"]["extraction_dims"] = (
            self.sph[slice_key]["params"]["patch_dims"][0]
            + 2 * self.sph[slice_key]["params"]["context_dims"][0],
            self.sph[slice_key]["params"]["patch_dims"][1]
            + 2 * self.sph[slice_key]["params"]["context_dims"][1],
        )

        self.sph[slice_key]["params"]["stride_dims"] = (
            self.sph[slice_key]["params"]["patch_dims"][0]
            - (
                self.sph[slice_key]["params"]["overlap_dims"][0]
                + 2 * self.sph[slice_key]["params"]["context_dims"][0]
            ),
            self.sph[slice_key]["params"]["patch_dims"][1]
            - (
                self.sph[slice_key]["params"]["overlap_dims"][1]
                + 2 * self.sph[slice_key]["params"]["context_dims"][1]
            ),
        )

        self.sph[slice_key]["params"]["factor1"] = self.wsi.factor_mpp(
            self.sph[slice_key]["params"]["target_mpp"]
        )
        self.sph[slice_key]["params"]["level"] = self.wsi.get_level_for_downsample(
            self.sph[slice_key]["params"]["factor1"]
        )
        self.sph[slice_key]["params"]["level_dims"] = self.wsi.get_level_dimensions()[
            self.sph[slice_key]["params"]["level"]
        ]
        self.sph[slice_key]["params"]["mpp_at_level"] = (
            self.wsi.get_level_downsamples()[self.sph[slice_key]["params"]["level"]]
            * self.wsi.mpp
        )
        self.sph[slice_key]["params"]["factor2"] = self.wsi.factor_mpp(
            self.sph[slice_key]["params"]["mpp_at_level"]
        )
        self.sph[slice_key]["params"]["factor3"] = self.wsi.factor_mpp(
            target_mpp=self.sph[slice_key]["params"]["target_mpp"],
            source_mpp=self.sph[slice_key]["params"]["mpp_at_level"],
        )

        self.sph[slice_key]["params"]["extraction_dims_at_level"] = (
            round_to_nearest_even(
                self.sph[slice_key]["params"]["extraction_dims"][0]
                * self.sph[slice_key]["params"]["factor3"]
            ),
            round_to_nearest_even(
                self.sph[slice_key]["params"]["extraction_dims"][1]
                * self.sph[slice_key]["params"]["factor3"]
            ),
        )

        self.sph[slice_key]["params"]["stride_dims_at_level"] = (
            round_to_nearest_even(
                self.sph[slice_key]["params"]["stride_dims"][0]
                * self.sph[slice_key]["params"]["factor3"]
            ),
            round_to_nearest_even(
                self.sph[slice_key]["params"]["stride_dims"][1]
                * self.sph[slice_key]["params"]["factor3"]
            ),
        )
    def get_slice_region(self, idx, slice_key):
        region = self.wsi.get_region(
            self.sph[self.slice_key]["coordinates"][idx][0],
            self.sph[self.slice_key]["coordinates"][idx][1],
            self.sph[slice_key]["params"]["extraction_dims_at_level"][0],
            self.sph[slice_key]["params"]["extraction_dims_at_level"][1],
            self.sph[self.slice_key]["params"]["level"],
        )
        region = array(region)
        region = resize(
            region,
            (
                self.sph[slice_key]["params"]["extraction_dims"][1],
                self.sph[slice_key]["params"]["extraction_dims"][0],
            ),
        )
        return region
