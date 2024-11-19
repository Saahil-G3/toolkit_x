from pathlib import Path
from typing import Union, List, Optional

from tiffslide import TiffSlide

from toolkit.geometry.shapely_tools import Polygon, MultiPolygon

from ._init_wsi import InitWSI


class TiffSlideWSI(InitWSI):
    def __init__(
        self, wsi_path: Path, tissue_geom: Union[Polygon, MultiPolygon] = None
    ):
        InitWSI.__init__(self, tissue_geom)

        self.wsi_type = "TiffSlide"
        self._wsi_path = Path(wsi_path)
        self._wsi = TiffSlide(self._wsi_path)
        self.dims = self._wsi.dimensions
        self._mpp_x = self._wsi.properties.get("tiffslide.mpp-x")
        self._mpp_y = self._wsi.properties.get("tiffslide.mpp-y")
        self.mpp = self._mpp_x
        if self._mpp_x != self._mpp_y:
            warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)

        self.level_count = self._wsi.level_count
        self._set_level_mpp_dict()

    def get_thumbnail_at_mpp(self, target_mpp=50):
        dims = self.get_dims_at_mpp(target_mpp)
        return self.get_thumbnail_at_dims(dims)

    def get_thumbnail_at_dims(self, dims):
        return self._wsi.get_thumbnail(dims)

    def get_level_for_downsample(self, factor):
        return self._wsi.get_best_level_for_downsample(factor)

    def get_region_for_slicer(self, coordinates, slice_params):
        x , y = coordinates
        w, h = slice_params["extraction_dims_at_level"]
        level = slice_params["level"]
        region = self._get_region(x, y, w, h, level).convert("RGB")
        region = region.resize(slice_params["extraction_dims"])
        return region

    def _get_region(self, x, y, w, h, level):
        return self._wsi.read_region(
            (int(x), int(y)),
            level,
            (int(w), int(h)),
        )

    def _set_level_mpp_dict(self):
        level_mpp_dict = {}
        for idx, factor in enumerate(self._wsi.level_downsamples):
            temp_dict = {}
            temp_dict["level"] = idx
            temp_dict["mpp"] = self.mpp * factor
            temp_dict["factor"] = self.factor_mpp(temp_dict["mpp"])
            temp_dict["dims"] = (
                int(self.dims[0] // temp_dict["factor"]),
                int(self.dims[1] // temp_dict["factor"]),
            )
            level_mpp_dict[idx] = temp_dict

        self.level_mpp_dict = level_mpp_dict

    def _get_slice_wsi_coordinates(self, slice_params):

        coordinates = []

        x_lim, y_lim = slice_params["level_dims"]
        extraction_dims_at_level = slice_params["extraction_dims_at_level"]
        stride_dims_at_level = slice_params["stride_dims_at_level"]
        context_dims = slice_params["context_dims"]
        factor2 = slice_params["factor2"]

        max_x = x_lim + context_dims[0]
        max_y = y_lim + context_dims[1]

        scaled_stride_x = stride_dims_at_level[0] * factor2
        scaled_stride_y = stride_dims_at_level[1] * factor2
        scaled_extraction_x = extraction_dims_at_level[0] * factor2
        scaled_extraction_y = extraction_dims_at_level[1] * factor2

        max_x_adj = max_x - extraction_dims_at_level[0]
        max_y_adj = max_y - extraction_dims_at_level[1]

        for x in range(-context_dims[0], max_x, stride_dims_at_level[0]):
            x_clipped = min(x, max_x_adj)
            x_scaled = int(self.round_to_nearest_even(x_clipped * factor2))

            for y in range(-context_dims[1], max_y, stride_dims_at_level[1]):
                y_clipped = min(y, max_y_adj)
                y_scaled = int(self.round_to_nearest_even(y_clipped * factor2))

                coordinates.append(((x_scaled, y_scaled), False))

        return coordinates
