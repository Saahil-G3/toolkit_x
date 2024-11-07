from ._tiffslide import TiffSlideWSI

class WSIManager():
    def __init__(self, wsi_path, wsi_type='TiffSlide'):
        self.wsi_type = wsi_type
        self.wsi_path = wsi_path
    def get_wsi_object(self, wsi_type):
        if wsi_type=='TiffSlide':
            return TiffSlideWSI(self.wsi_path)

        
        