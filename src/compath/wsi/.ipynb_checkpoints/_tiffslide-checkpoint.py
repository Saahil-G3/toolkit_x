from _init_wsi import InitWSI
from tiffslide import TiffSlide

class TiffSlideWSI(InitWSI):
    def __init__(self, wsi_path):
        super(TiffSlideWSI, self).__init__()
        
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
