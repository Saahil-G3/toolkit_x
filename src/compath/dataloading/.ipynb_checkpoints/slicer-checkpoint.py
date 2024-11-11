import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

#mp.set_start_method('spawn', force=True)

from geometry.shapely_tools import loads, get_box, flatten_geom_collection
from geometry.geomtorch import fill_polygon, get_polygon_coordinates, pil_to_tensor, resize, median_blur

from ._init_slicer import InitSlicer
from ._torch_datasets import _DatasetAll, _DatasetContained, _DatasetBoundary

class Slicer(InitSlicer):
    def __init__(self, wsi, device, tissue_geom = None):
        InitSlicer.__init__(self, wsi, tissue_geom = None)
        self.slice_key = None
        self.device = device
        
    def set_tissue_geom(self, tissue_geom):
        """
        Overriding the parent's set_tissue_geom method, if needed.
        """
        super().set_tissue_geom(tissue_geom)
    
    def get_torch_dataloader(self, loading_type, batch_size=2, shuffle=False, num_workers=0, **kwargs):
        
        if loading_type=='all':
            dataset = _DatasetAll(self)
            self.n_samples = len(self.sph[self.slice_key]["coordinates"])
        elif loading_type=='contained':
            dataset = _DatasetContained(self)
            self.n_samples_contained = len(self.sph[self.slice_key]["filtered_coordinates"]["contained_coordinates"])
        elif loading_type=='boundary':
            dataset = _DatasetBoundary(self)
            self.n_samples_boundary = len(self.sph[self.slice_key]["filtered_coordinates"]["boundary_coordinates"])
        else:
            raise ValueError(f"loading type {loading_type} not implemented")

        if self.wsi.wsi_type == 'TiffSlide' and num_workers>0:
            dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    worker_init_fn=dataset.worker_init,
                    **kwargs,
                )
        else:
            dataloader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    **kwargs,
                )
        return dataloader


    def set_slice_key(self, slice_key):
        self.slice_key = slice_key

    """
    def _get_region_from_contained_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]['filtered_coordinates']['contained_coordinates'][idx]
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"])
        return region
        
    def _get_region_from_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]['coordinates'][idx]
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"])
        return region
        
    def _get_region_from_boundary_coordinates(self, idx):
        coordinates = self.sph[self.slice_key]["filtered_coordinates"]["boundary_coordinates"][idx]
        scale_factor = 1 / self.params["factor1"]
        mask_dims = self.params["extraction_dims"]
    
        region = self.get_slice_region(self.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.params["extraction_dims"])
    
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
    
        #mask = torch.zeros(mask_dims, dtype=torch.uint8, device=self.device)
        exterior_masks = torch.stack(
            [fill_polygon(polygon, mask_dims, device=self.device) for polygon in exterior]
        )
        #mask += exterior_masks.sum(dim=0)
        mask = exterior_masks.sum(dim=0)
        
        if len(holes)>0:
            hole_masks = torch.stack(
                [fill_polygon(polygon, mask_dims, device=self.device) for polygon in holes]
            )
            mask -= hole_masks.sum(dim=0)
            
        return region, mask

    """
    def apply_median_blur(self, mask, median_blur_kernel = 15):
        num_dims = mask.dim()
    
        if num_dims == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif num_dims == 3:
            mask = mask.unsqueeze(0)
        elif num_dims > 3:
            raise ValueError("Mask has too many dimensions to apply median_blur.")
    
        mask = median_blur(mask, median_blur_kernel)
        
        return mask