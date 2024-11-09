import torch
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader

from geometry.shapely_tools import loads, get_box, flatten_geom_collection
from geometry.geomtorch import fill_polygon, get_polygon_coordinates, pil_to_tensor, resize
from ._init_slicer import InitSlicer

class Slicer(BaseDataset, InitSlicer):
    def __init__(self, wsi, device, tissue_geom = None):
        InitSlicer.__init__(self, wsi, tissue_geom = None)
        self.slice_key = None
        self.device = device
        self.merge_coordinates = []
        
    def set_tissue_geom(self, tissue_geom):
        """
        Overriding the parent's set_tissue_geom method, if needed.
        """
        super().set_tissue_geom(tissue_geom)

    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        if self.sample_using_tissue_geom:
            self.filtered_index += 1
            if idx < self.n_contained_coordinates:
                return self._get_region_from_contained_coordinates(self.filtered_index)
            else:
                return self._get_region_from_boundary_coordinates(
                    self.filtered_index - self.n_contained_coordinates
                )
        else:
            return self._get_region_from_coordinates(idx)

                
    def get_dataloader(
        self,
        batch_size=4,
        shuffle=False,
        **kwargs,
    ):
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        return dataloader


    def set_slice_key(self, slice_key):
        self.slice_key = slice_key
            
    def _get_region_from_contained_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]['filtered_coordinates']['contained_coordinates'][idx]
        self.merge_coordinates.append(coordinates)
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"]).to(self.device)
        return region
        
    def _get_region_from_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]['coordinates'][idx]
        self.merge_coordinates.append(coordinates)
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"]).to(self.device)
        return region
        
    def _get_region_from_boundary_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]["filtered_coordinates"]["boundary_coordinates"][idx]
        self.merge_coordinates.append(coordinates)
        scale_factor = 1 / self.params["factor1"]
        mask_dims = self.params["extraction_dims"]
    
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"]).to(self.device)
    
        box = get_box(
            coordinates[0],
            coordinates[1],
            self.params["extraction_dims"][0] * self.params["factor1"],
            self.params["extraction_dims"][1] * self.params["factor1"],
        )
        geom_region = self.tissue_geom.intersection(box)
    
        origin = torch.tensor([coordinates[0], coordinates[1]], dtype=torch.float32, device=self.device)
        geom_dict = flatten_geom_collection(geom_region)
    
        exterior, holes = [], []
    
        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates(
                polygon, scale_factor=scale_factor, origin=origin, device=self.device
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])
    
        mask = torch.zeros(mask_dims, dtype=torch.uint8, device=self.device)
    
        for polygon in exterior:
            mask += fill_polygon(polygon, mask_dims, device=self.device)
    
        for polygon in holes:
            mask -= fill_polygon(polygon, mask_dims, device=self.device)
    
        region *= mask
    
        return region