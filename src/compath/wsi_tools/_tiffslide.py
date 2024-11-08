from tiffslide import TiffSlide

from ._init_wsi import InitWSI

class TiffSlideWSI(InitWSI):
    def __init__(self, wsi_path, tissue_geom=None):
        InitWSI.__init__(self, tissue_geom)
        
        self.wsi_type = 'TiffSlide'
        self._wsi_path = wsi_path
        self._wsi = TiffSlide(self._wsi_path)
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
        
                coordinates.append((x_scaled, y_scaled))
    
        return coordinates

