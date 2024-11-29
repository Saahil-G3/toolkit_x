import cv2
import torch
import warnings
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from toolkit.geometry.torch_tools import (
    resize,
    fill_polygon,
    pil_to_tensor,
)
from toolkit.geometry.shapely_tools import (
    get_box,
    flatten_geom_collection,
    get_polygon_coordinates_cpu,
    #get_polygon_coordinates_gpu,
)
from ._init_slicer import _InitSlicer
from toolkit.pathomics.slide._tiffslide import TiffSlideWSI
from toolkit.vision.deep_learning.torchmodel import _BaseModel

def extract_patches_with_coordinates(image, patch_dims, overlap_dims):
    """
    Extract patches from a large numpy array with specified patch size and overlap.

    Parameters:
    - image: numpy array, the input image
    - patch_size: tuple, (h, w), size of the patches to be extracted
    - overlap: tuple, (oh, ow), overlap in the vertical and horizontal directions

    Returns:
    - patches: list, a list of extracted patches
    - coordinates: list, a list of tuples containing the (start_row, start_col) coordinates of each patch
    """

    h, w = patch_dims
    oh, ow = overlap_dims

    patches = []
    coordinates = []

    flag = 0

    print("Extracting Patches")
    for i in range(0, image.shape[0], h - oh):
        if i + h > image.shape[0]:
            i = image.shape[0] - h

        for j in range(0, image.shape[1], w - ow):
            if j + w > image.shape[1]:
                j = image.shape[1] - w

            patch = image[i : i + h, j : j + w, :]

            patches.append(patch)
            coordinates.append((i, j))
            flag += 1

    print(f"Complete!")
    print(f"Extracted {flag} patches.")

    return np.array(patches), coordinates

class Slicer(_InitSlicer):
    def __init__(
        self,
        gpu_id: int = 0,
        device_type: str = "gpu",
        results_path: Path = None,
        dataparallel: bool = False,
        dataparallel_device_ids=None,
    ):
        _InitSlicer.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            results_path=results_path,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.slice_key = None

    # def get_inference_dataloader(
    #     self,
    #     batch_size=2,
    #     shuffle=False,
    #     num_workers=0,
    #     **kwargs,
    # ):
    #     """
    #     Creates a PyTorch DataLoader for inference based on specified coordinate types.

    #     Parameters:
    #     - `batch_size` (int, optional): Number of samples per batch to load. Default is 2.
    #     - `shuffle` (bool, optional): Whether to shuffle the data at every epoch. Default is `False`.
    #     - `num_workers` (int, optional): Number of subprocesses to use for data loading. If set to 0, data will be loaded in the main process. Default is 0.
    #     - `**kwargs`: Additional arguments for `DataLoader`.

    #     Returns:
    #     - `DataLoader`: A PyTorch DataLoader for inference.

    #     Notes:
    #     - If the whole slide image type (`wsi.wsi_type`) is `"TiffSlide"` and `num_workers` is greater than 0, a custom worker initialization function `_worker_init_tiffslide` is used.
    #     """
    #     dataset = InferenceDataset(self)

    #     if self.wsi.wsi_type == "TiffSlide" and num_workers > 0:
    #         dataloader = DataLoader(
    #             dataset,
    #             batch_size=batch_size,
    #             shuffle=shuffle,
    #             num_workers=num_workers,
    #             worker_init_fn=self._worker_init_tiffslide,
    #             **kwargs,
    #         )

    #     else:
    #         dataloader = DataLoader(
    #             dataset,
    #             batch_size=batch_size,
    #             shuffle=shuffle,
    #             num_workers=num_workers,
    #             **kwargs,
    #         )
    #     return dataloader

    #def _worker_init_tiffslide(self, *args):
    #    self.wsi = TiffSlideWSI(
    #        wsi_path=self.wsi._wsi_path, tissue_geom=self.wsi.tissue_geom
    #    )

    #def _set_slice_key(self, slice_key):
    #    self.slice_key = slice_key

    #def _get_region_mask(self, coordinate, params):
    #    origin = np.array([coordinate[0], coordinate[1]], dtype=np.float32)
    #    mask_dims = params["extraction_dims"]
    #    scale_factor = 1 / params["factor1"]

    #   box = get_box(
    #        coordinate[0],
    #        coordinate[1],
    #        params["extraction_dims"][0] * params["factor1"],
    #        params["extraction_dims"][1] * params["factor1"],
    #    )
    #    geom_region = self.tissue_geom.intersection(box).buffer(0)

    #    if geom_region.area == 0:
    #        mask = np.zeros(mask_dims, dtype=np.uint8)
    #        return mask

    #    geom_dict = flatten_geom_collection(geom_region)

    #    if len(geom_dict) > 1:
    #        warnings.warn(
    #            f"Multiple geometries detected in tissue mask. Check: {', '.join(geom_dict.keys())}"
    #        )

    #    exterior, holes = [], []
    #    for polygon in geom_dict["Polygon"]:
    #        polygon_coordinates = get_polygon_coordinates_cpu(
    #            polygon, scale_factor=scale_factor, origin=origin
    #        )
    #        exterior.extend(polygon_coordinates[0])
    #        holes.extend(polygon_coordinates[1])

    #   mask = np.zeros(mask_dims, dtype=np.uint8)

    #    # for polygon in exterior:
    #    #    cv2.fillPoly(mask, [polygon], 1)
    #    cv2.fillPoly(mask, exterior, 1)

    #    if len(holes) > 0:
    #        # for polygon in holes:
    #        #    cv2.fillPoly(mask, [polygon], 0)
    #        cv2.fillPoly(mask, holes, 0)

    #    return mask

    # def _get_region_mask_gpu(self, coordinate, params):
    #    origin = torch.tensor(
    #        [coordinate[0], coordinate[1]],
    #        dtype=torch.float32,
    #        device=self.device,
    #    )
    #    mask_dims = params["extraction_dims"]
    #    scale_factor = 1 / params["factor1"]

    #   box = get_box(
    #        coordinate[0],
    #        coordinate[1],
    #        params["extraction_dims"][0] * params["factor1"],
    #        params["extraction_dims"][1] * params["factor1"],
    #    )
    #    geom_region = self.tissue_geom.intersection(box).buffer(0)
    #    geom_dict = flatten_geom_collection(geom_region)

    #    if len(geom_dict) > 1:
    #        logger.warning("Multiple geometry detected in tissue mask, check")

    #    exterior, holes = [], []
    #    for polygon in geom_dict["Polygon"]:
    #        polygon_coordinates = get_polygon_coordinates_gpu(
    #            polygon,
    #            scale_factor=scale_factor,
    #            origin=origin,
    #            device=self.device,
    #        )
    #        exterior.extend(polygon_coordinates[0])
    #        holes.extend(polygon_coordinates[1])

    #    exterior_masks = torch.stack(
    #        [
    #            fill_polygon(polygon, mask_dims, device=self.device)
    #            for polygon in exterior
    #        ]
    #    )
    #    mask = exterior_masks.sum(dim=0)

    #    if len(holes) > 0:
    #        hole_masks = torch.stack(
    #            [
    #                fill_polygon(polygon, mask_dims, device=self.device)
    #                for polygon in holes
    #            ]
    #        )
    #        mask -= hole_masks.sum(dim=0)
    #    return mask


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
