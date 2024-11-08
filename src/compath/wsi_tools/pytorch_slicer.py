from cv2 import resize
from numpy import array

from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

from ._init_slicer import InitSlicer

class Slicer(BaseDataset, InitSlicer):
    def __init__(self, wsi, tissue_geom = None):
        InitSlicer.__init__(self, wsi, tissue_geom = None)
        self.slice_key = None
        
    def set_tissue_geom(self, tissue_geom):
        """
        Overriding the parent's set_tissue_geom method, if needed.
        You can call the parent's method to set tissue_geom in Slicer, 
        and add any additional logic specific to Diagnosis.
        """
        super().set_tissue_geom(tissue_geom)

    def set_slice_key(self, slice_key):
        self.slice_key = slice_key

    def __getitem__(self, idx):
        params = self.sph[self.slice_key]["params"]
        coordinates = self.sph[self.slice_key]["coordinates"][idx]
        
        region = self.get_slice_region(params, coordinates)
        region = resize(
            array(region),
            params["extraction_dims"],
            )
        region = region.transpose(2, 0, 1)
        return region

    def __len__(self):
        if not self.set_params_init:
            raise ValueError("No slice parameters have been initialized")

        if self.slice_key is None:
            return len(self.sph[self.recent_slice_key]["coordinates"])
        else:
            return len(self.sph[self.slice_key]["coordinates"])

    def get_dataloader(
        self,
        #dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        **kwargs,
    ):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **kwargs,
        )
        return dataloader
