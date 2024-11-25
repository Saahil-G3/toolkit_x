
from torch.utils.data import Dataset as BaseDataset

from toolkit.geometry.torch_tools import (
    resize,
    pil_to_tensor,
)

class InferenceDataset(BaseDataset):
    def __init__(self, slicer):
        self.slicer = slicer
        self.coordinates = self.slicer.sph[self.slicer.slice_key][
            self.slicer._coordinates_type
        ]
        self.params = self.slicer.sph[self.slicer.slice_key]["params"]
        self.wsi_name = self.slicer.sph[self.slicer.slice_key]["wsi_name"]
        self.device = self.slicer.device

    def __len__(self):
        return len(self.coordinates)

    #def __getitem__(self, idx):
    #    coordinate, is_boundary = self.coordinates[idx]
    #    region = pil_to_tensor(self.slicer.get_slice_region(coordinate, self.params))
    #    region = resize(region, self.params["extraction_dims"])

    #    if is_boundary:
    #        mask = self.slicer._get_region_mask(coordinate, self.params)
    #        mask = torch.from_numpy(mask)
    #    else:
    #        mask = torch.ones(self.params["extraction_dims"], dtype=torch.uint8)

    #    return region, mask

    def __getitem__(self, idx):
        coordinate, is_boundary = self.coordinates[idx]
        region = pil_to_tensor(self.slicer.get_slice_region(coordinate, self.params))
        region = resize(region, self.params["extraction_dims"])
        
        return region