
class UnitConverter():
    def __init__(self):
        pass

    def get_area_in_microns2(self, mpp, pixel2_area):
        microns2_area = round(pixel2_area * (mpp ** 2), 2)
        return microns2_area
    
    def get_length_in_microns(self, mpp, pixel_length):
        microns_length = round(pixel_length * (mpp), 2)
        return microns_length
    
    def get_area_in_mm2(self, mpp, pixel2_area):
        mm2_area = round(pixel2_area * (mpp ** 2) / 1_000_000, 2)
        return mm2_area
    
    def get_length_in_mm(self, mpp, pixel_length):
        mm_length = round(pixel_length * (mpp) / 1_000, 2)
        return mm_length