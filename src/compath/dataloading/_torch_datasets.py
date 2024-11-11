import torch
from compath.slide._tiffslide import TiffSlideWSI
#from tiffslide import TiffSlide
from torch.utils.data import Dataset as BaseDataset
from geometry.shapely_tools import loads, get_box, flatten_geom_collection
from geometry.geomtorch import fill_polygon, get_polygon_coordinates, pil_to_tensor, resize, median_blur

class _CPathDataset(BaseDataset):
    def __init__(self, slicer):
        self.slicer = slicer

    def worker_init(self, *args):
        self.slicer.wsi = TiffSlideWSI(wsi_path = self.slicer.wsi._wsi_path, tissue_geom=self.slicer.wsi.tissue_geom)

class _DatasetAll(_CPathDataset):
    def __init__(self, slicer):
        _CPathDataset.__init__(self, slicer)
    
    def __len__(self):
        return self.slicer.n_samples

    def __getitem__(self, idx):
        coordinates = self.slicer.sph[self.slicer.slice_key]['coordinates'][idx]
        region = self.slicer.get_slice_region(self.slicer.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.slicer.params["extraction_dims"])
        return region
        #return self.slicer._get_region_from_coordinates(idx)
        
class _DatasetContained(_CPathDataset):
    def __init__(self, slicer):
        _CPathDataset.__init__(self, slicer)
    
    def __len__(self):
        return self.slicer.n_samples_contained

    def __getitem__(self, idx):
        
        coordinates = self.slicer.sph[self.slicer.slice_key]['filtered_coordinates']['contained_coordinates'][idx]
        region = self.slicer.get_slice_region(self.slicer.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.slicer.params["extraction_dims"])
        return region
        #return self.slicer._get_region_from_contained_coordinates(idx)

class _DatasetBoundary(_CPathDataset):
    def __init__(self, slicer):
        _CPathDataset.__init__(self, slicer)
    
    def __len__(self):
        return self.slicer.n_samples_boundary

    def __getitem__(self, idx):
        coordinates = self.slicer.sph[self.slicer.slice_key]["filtered_coordinates"]["boundary_coordinates"][idx]
        scale_factor = 1 / self.slicer.params["factor1"]
        mask_dims = self.slicer.params["extraction_dims"]
    
        region = self.slicer.get_slice_region(self.slicer.params, coordinates)
        region = pil_to_tensor(region)
        region = resize(region, self.slicer.params["extraction_dims"])
    
        box = get_box(
            coordinates[0],
            coordinates[1],
            self.slicer.params["extraction_dims"][0] * self.slicer.params["factor1"],
            self.slicer.params["extraction_dims"][1] * self.slicer.params["factor1"],
        )
        geom_region = self.slicer.tissue_geom.intersection(box)
    
        origin = torch.tensor([coordinates[0], coordinates[1]], dtype=torch.float32, device=self.slicer.device)
        geom_dict = flatten_geom_collection(geom_region)
    
        exterior, holes = [], []
    
        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates(
                polygon, scale_factor=scale_factor, origin=origin, device=self.slicer.device
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])
    
        exterior_masks = torch.stack(
            [fill_polygon(polygon, mask_dims, device=self.slicer.device) for polygon in exterior]
        )
        mask = exterior_masks.sum(dim=0)
        
        if len(holes)>0:
            hole_masks = torch.stack(
                [fill_polygon(polygon, mask_dims, device=self.slicer.device) for polygon in holes]
            )
            mask -= hole_masks.sum(dim=0)
            
        return region, mask
        #return self.slicer._get_region_from_boundary_coordinates(idx)



