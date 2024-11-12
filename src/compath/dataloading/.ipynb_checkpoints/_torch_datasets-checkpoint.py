import cv2
import torch
import numpy as np
from compath.slide._tiffslide import TiffSlideWSI
from torch.utils.data import Dataset as BaseDataset
from geometry.shapely_tools import (
    loads,
    get_box,
    flatten_geom_collection,
    get_polygon_coordinates_cpu,
    get_polygon_coordinates_gpu,
)
from geometry.geomtorch import (
    fill_polygon,
    pil_to_tensor,
    resize,
    median_blur,
)


class _CPathDataset(BaseDataset):
    def __init__(self, slicer, data_loading_mode="cpu"):
        self.slicer = slicer
        self.data_loading_mode = data_loading_mode

    def worker_init(self, *args):
        self.slicer.wsi = TiffSlideWSI(
            wsi_path=self.slicer.wsi._wsi_path, tissue_geom=self.slicer.wsi.tissue_geom
        )

    def get_region_mask_cpu(self, coordinate):
        origin = np.array([coordinate[0], coordinate[1]], dtype=np.float32)
        mask_dims = self.slicer.params["extraction_dims"]
        scale_factor = 1 / self.slicer.params["factor1"]
        
        box = get_box(
            coordinate[0],
            coordinate[1],
            self.slicer.params["extraction_dims"][0] * self.slicer.params["factor1"],
            self.slicer.params["extraction_dims"][1] * self.slicer.params["factor1"],
        )
        geom_region = self.slicer.tissue_geom.intersection(box)
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
            device=self.slicer.device,
        )
        mask_dims = self.slicer.params["extraction_dims"]
        scale_factor = 1 / self.slicer.params["factor1"]
        
        box = get_box(
            coordinate[0],
            coordinate[1],
            self.slicer.params["extraction_dims"][0] * self.slicer.params["factor1"],
            self.slicer.params["extraction_dims"][1] * self.slicer.params["factor1"],
        )
        geom_region = self.slicer.tissue_geom.intersection(box)
        geom_dict = flatten_geom_collection(geom_region)

        exterior, holes = [], []

        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates_gpu(
                polygon,
                scale_factor=scale_factor,
                origin=origin,
                device=self.slicer.device,
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])

        exterior_masks = torch.stack(
            [
                fill_polygon(polygon, mask_dims, device=self.slicer.device)
                for polygon in exterior
            ]
        )
        mask = exterior_masks.sum(dim=0)

        if len(holes) > 0:
            hole_masks = torch.stack(
                [
                    fill_polygon(polygon, mask_dims, device=self.slicer.device)
                    for polygon in holes
                ]
            )
            mask -= hole_masks.sum(dim=0)

        return mask


class _DatasetAll(_CPathDataset):
    def __init__(self, slicer, data_loading_mode="cpu"):
        _CPathDataset.__init__(self, slicer, data_loading_mode)

    def __len__(self):
        return len(self.slicer.sph[self.slicer.slice_key]["all_coordinates"])

    def __getitem__(self, idx):
        coordinate = self.slicer.sph[self.slicer.slice_key]["all_coordinates"][idx]
        region = self.slicer.get_slice_region(self.slicer.params, coordinate)
        region = pil_to_tensor(region)
        region = resize(region, self.slicer.params["extraction_dims"])
        return region


class _DatasetFiltered(_CPathDataset):
    def __init__(self, slicer, data_loading_mode="cpu"):
        _CPathDataset.__init__(self, slicer, data_loading_mode)

    def __len__(self):
        return len(self.slicer.sph[self.slicer.slice_key]["filtered_coordinates"])

    def __getitem__(self, idx):

        coordinate, is_boundary = self.slicer.sph[self.slicer.slice_key][
            "filtered_coordinates"
        ][idx]
        region = self.slicer.get_slice_region(self.slicer.params, coordinate)
        region = pil_to_tensor(region)
        region = resize(region, self.slicer.params["extraction_dims"])
        
        if is_boundary:
            if self.data_loading_mode == "cpu":
                mask = torch.from_numpy(self.get_region_mask_cpu(coordinate))
            elif self.data_loading_mode == "gpu":
                mask = self.get_region_mask_gpu(coordinate)
            else:
                raise ValueError(
                    f"loading mode {self.data_loading_mode} not implemented, choose between 'cpu' or 'gpu'"
                )
        else:
            mask = torch.ones(
                self.slicer.params["extraction_dims"],
                dtype=torch.uint8,
                device=self.slicer.device if self.data_loading_mode == "gpu" else "cpu",
            )


        return region, mask
