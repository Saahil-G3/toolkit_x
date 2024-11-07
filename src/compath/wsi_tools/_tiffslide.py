from tiffslide import TiffSlide

from ._init_wsi import InitWSI
from misc import round_to_nearest_even

class TiffSlideWSI(InitWSI):
    def __init__(self, wsi_path, tissue_geom=None):
        InitWSI.__init__(self, tissue_geom)
        
        self.wsi_type = 'TiffSlide'
        self._wsi = TiffSlide(wsi_path)
        self.dims = self._wsi.dimensions
        self.level_count = self.get_level_count()
        self._mpp_x = self._wsi.properties.get('tiffslide.mpp-x')
        self._mpp_y = self._wsi.properties.get('tiffslide.mpp-y')
        self.mpp = self._mpp_x
        if self._mpp_x != self._mpp_y:
            warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)

    def get_thumbnail_at_mpp(self, target_mpp=50):
        return self._wsi.get_thumbnail(self.get_dims_at_mpp(target_mpp))

    def get_thumbnail_at_dims(self, dims):
        return self._wsi.get_thumbnail(dims)

    def get_region(self, x, y, w, h, level):
        return self._wsi.read_region(
            (int(x), int(y)),
            level,
            (int(w), int(h)),
        )

    def get_level_for_downsample(self, factor):
        return self._wsi.get_best_level_for_downsample(factor)

    def get_level_dimensions(self):
        return self._wsi.level_dimensions

    def get_level_downsamples(self):
        return self._wsi.level_downsamples

    def get_level_count(self):
        return self._wsi.level_count

    def _get_slice_wsi_coordinates(self, slice_params):
        
        coordinates = []
    
        x_lim, y_lim = slice_params["level_dims"]
        extraction_dims_at_level = slice_params["extraction_dims_at_level"]
        stride_dims_at_level = slice_params["stride_dims_at_level"]
        context_dims = slice_params["context_dims"]
        factor2 = slice_params["factor2"]
    
        for x in range(-context_dims[0], x_lim + context_dims[0], stride_dims_at_level[0]):
            if x + extraction_dims_at_level[0] > x_lim + context_dims[0]:
                x = (x_lim + context_dims[0]) - extraction_dims_at_level[0]
    
            for y in range(-context_dims[1], y_lim + context_dims[1], stride_dims_at_level[1]):
                if y + extraction_dims_at_level[1] > y_lim + context_dims[1]:
                    y = (y_lim + context_dims[1]) - extraction_dims_at_level[1]
    
                x_scaled, y_scaled = int(round_to_nearest_even(x * factor2)), int(
                    round_to_nearest_even(y * factor2)
                )
    
                coordinates.append((x_scaled, y_scaled))
    
        return coordinates

