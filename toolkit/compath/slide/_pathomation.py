from pathlib import Path
from typing import Union, List, Optional

from pma_python import core

from toolkit.geometry.shapely_tools import Polygon, MultiPolygon

from ._init_wsi import InitWSI


class PathomationWSI(InitWSI):
    def __init__(
        self,
        wsi_path: Path,
        sessionID: str = None,
        tissue_geom: Union[Polygon, MultiPolygon] = None,
    ):
        InitWSI.__init__(self, tissue_geom)

        self.sessionID = sessionID
        self.wsi_type = "Pathomation"
        self._wsi_path = Path(wsi_path)
        self._slideRef = str(self._wsi_path)
        self.dims = core.get_pixel_dimensions(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self._mpp_x, self._mpp_y = core.get_pixels_per_micrometer(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self.mpp = self._mpp_x

        if self._mpp_x != self._mpp_y:
            warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)

        self.zoomlevels = core.get_zoomlevels_list(
            self._slideRef, sessionID=self.sessionID, min_number_of_tiles=0
        )
        self.level_count = len(self.zoomlevels)
        self._set_level_mpp_dict()

    def get_thumbnail_at_mpp(self, target_mpp=50):
        factor = self.factor_mpp(target_mpp)
        dims = (int(self.dims[0] // factor), int(self.dims[1] // factor))
        return self.get_thumbnail_at_dims(dims)

    def get_thumbnail_at_dims(self, dims):
        thumbnail = core.get_thumbnail_image(
            self._slideRef,
            width=dims[0],
            height=dims[1],
            sessionID=self.sessionID,
            verify=True,
        )
        return thumbnail

    def get_level_for_downsample(self, factor):
        for key, value in self.level_mpp_dict.items():
            if value["factor"]<factor:
                break
        return key

    def get_region_for_slicer(self, coordinate, slice_params):
        x , y = coordinate
        w, h = slice_params['extraction_dims']
        factor = slice_params['factor1']
        w_scaled = self.round_to_nearest_even(w*factor)
        h_scaled = self.round_to_nearest_even(h*factor)
        region = self._get_region(x, y, w_scaled, h_scaled, scale=1/factor).convert("RGB")
        return region

    def _get_region(self, x: int, y: int, w: int, h: int, scale: float = 1):
        region = core.get_region(
            self._slideRef,
            x=x,
            y=y,
            width=w,
            height=h,
            scale=scale,
            sessionID=self.sessionID,
        )
        return region


    def _set_level_mpp_dict(self):
        level_mpp_dict = {}
        for level in self.zoomlevels:
            temp_dict = {}
            mpp_x, mpp_y = core.get_pixels_per_micrometer(
                self._slideRef, zoomlevel=level, sessionID=self.sessionID
            )

            temp_dict["level"] = level
            temp_dict["mpp"] = mpp_x
            #factor to go from original mpp to mpp_x
            temp_dict["factor"] = self.factor_mpp(temp_dict["mpp"])
            temp_dict["dims"] = (
                int(self.dims[0]//temp_dict["factor"]), 
                int(self.dims[1]//temp_dict["factor"])
            )
            level_mpp_dict[self.level_count - level - 1] = temp_dict
            if mpp_x != mpp_y:
                warnings.warn(
                    f"mpp_x is not equal to mpp_y at level {level}", UserWarning
                )

        self.level_mpp_dict = level_mpp_dict

    def _get_slice_wsi_coordinates(self, slice_params):
        coordinates = []
        factor1 = slice_params["factor1"]
        x_lim, y_lim = self.dims
        extraction_dims = slice_params["extraction_dims"]
        stride_dims = slice_params["stride_dims"]
        context_dims = slice_params["context_dims"]
        
        scaled_extraction_dims = int(extraction_dims[0]*factor1), int(extraction_dims[1]*factor1)
        scaled_stride_dims = int(stride_dims[0]*factor1), int(stride_dims[1]*factor1)
        scaled_context_dims = int(context_dims[0]*factor1), int(context_dims[1]*factor1)
        
        max_x = x_lim + scaled_context_dims[0]
        max_y = y_lim + scaled_context_dims[1]
        
        max_x_adj = max_x - scaled_extraction_dims[0]
        max_y_adj = max_y - scaled_extraction_dims[1]
        
        for x in range(-scaled_context_dims[0], max_x, scaled_stride_dims[0]):
            x_clipped = min(x, max_x_adj)
        
            for y in range(-scaled_context_dims[1], max_y, scaled_stride_dims[1]):
                y_clipped = min(y, max_y_adj)
        
                coordinates.append(((x, y), False))
        return coordinates