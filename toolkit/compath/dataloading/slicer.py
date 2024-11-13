import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from toolkit.geometry.geomtorch import (
    resize,
    fill_polygon,
    pil_to_tensor,
)
from toolkit.geometry.shapely_tools import (
    loads,
    get_box,
    flatten_geom_collection,
    get_polygon_coordinates_cpu,
    get_polygon_coordinates_gpu,
)
from ._init_slicer import InitSlicer
from toolkit.compath.slide._tiffslide import TiffSlideWSI

class Slicer(InitSlicer):
    def __init__(self, wsi, device, tissue_geom=None, data_loading_mode="cpu"):
        InitSlicer.__init__(self, wsi, tissue_geom=None)
        self.data_loading_mode = data_loading_mode
        self.slice_key = None
        self.device = device

    def set_tissue_geom(self, tissue_geom):
        """
        Overriding the parent's set_tissue_geom method, if needed.
        """
        super().set_tissue_geom(tissue_geom)

    def get_inference_dataloader(
        self,
        coordinates_type,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        data_loading_mode="cpu",
        **kwargs,
    ):

        dataset = _InferenceDataset(self, coordinates_type, self.data_loading_mode)
    
        if self.wsi.wsi_type == "TiffSlide" and num_workers > 0:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=self._worker_init_tiffslide,
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

    def _worker_init_tiffslide(self, *args):
        self.wsi = TiffSlideWSI(
            wsi_path=self.wsi._wsi_path, tissue_geom=self.wsi.tissue_geom
        )

    def get_region_mask_cpu(self, coordinate):
        origin = np.array([coordinate[0], coordinate[1]], dtype=np.float32)
        mask_dims = self.params["extraction_dims"]
        scale_factor = 1 / self.params["factor1"]
        
        box = get_box(
            coordinate[0],
            coordinate[1],
            self.params["extraction_dims"][0] * self.params["factor1"],
            self.params["extraction_dims"][1] * self.params["factor1"],
        )
        geom_region = self.tissue_geom.intersection(box)
        geom_dict = flatten_geom_collection(geom_region)
        
        exterior, holes = [], []

        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates_cpu(
                polygon, scale_factor=scale_factor, origin=origin
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])

        mask = np.zeros(mask_dims, dtype=np.uint8)

        for polygon in exterior:
            cv2.fillPoly(mask, [polygon], 1)

        if len(holes) > 0:
            for polygon in holes:
                cv2.fillPoly(mask, [polygon], 0)

        return mask

    def get_region_mask_gpu(self, coordinate):
        origin = torch.tensor(
            [coordinate[0], coordinate[1]],
            dtype=torch.float32,
            device=self.device,
        )
        mask_dims = self.params["extraction_dims"]
        scale_factor = 1 / self.params["factor1"]
        
        box = get_box(
            coordinate[0],
            coordinate[1],
            self.params["extraction_dims"][0] * self.params["factor1"],
            self.params["extraction_dims"][1] * self.params["factor1"],
        )
        geom_region = self.tissue_geom.intersection(box)
        geom_dict = flatten_geom_collection(geom_region)

        exterior, holes = [], []

        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates_gpu(
                polygon,
                scale_factor=scale_factor,
                origin=origin,
                device=self.device,
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])

        exterior_masks = torch.stack(
            [
                fill_polygon(polygon, mask_dims, device=self.device)
                for polygon in exterior
            ]
        )
        mask = exterior_masks.sum(dim=0)

        if len(holes) > 0:
            hole_masks = torch.stack(
                [
                    fill_polygon(polygon, mask_dims, device=self.device)
                    for polygon in holes
                ]
            )
            mask -= hole_masks.sum(dim=0)

        return mask

class _InferenceDataset(BaseDataset):
    def __init__(
        self, slicer, coordinates_type="all_coordinates", data_loading_mode="cpu"
    ):
        self.data_loading_mode = data_loading_mode
        self.coordinates_type = coordinates_type
        self.coordinates = self.slicer.sph[self.slicer.slice_key][coordinates_type]
        self.params = self.slicer.sph[self.slicer.slice_key][params]
        self.device = self.slicer.device

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        coordinate, is_boundary = self.coordinates[idx]
        region = pil_to_tensor(
            self.slicer.get_slice_region(self.params, coordinate)
        )
        region = resize(region, self.extraction_dims)

        if is_boundary:
            mask = self._get_boundary_mask(coordinate)
        else:
            mask = torch.ones(
                self.extraction_dims, dtype=torch.uint8, device=self.device
            )

        return region, mask

    def _get_boundary_mask(self, coordinate):

        if self.data_loading_mode == "cpu":
            return torch.from_numpy(self.slicer.get_region_mask_cpu(coordinate))
        elif self.data_loading_mode == "gpu":
            return self.slicer.get_region_mask_gpu(coordinate)
        else:
            raise ValueError(
                f"Loading mode {self.data_loading_mode} not implemented, choose 'cpu' or 'gpu'"
            )