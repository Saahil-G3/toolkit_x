import cv2
import warnings
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from typing import Union, List, Optional

from torch.utils.data import DataLoader

from toolkit.geometry.shapely_tools import (
    loads,
    get_box,
    prep_geom_for_query,
    flatten_geom_collection,
    get_polygon_coordinates_cpu,
)

from toolkit.pathomics.wsi.manager import WSIManager
from toolkit.pathomics.wsi.tiffslide import TiffSlideWSI
from toolkit.vision.deep_learning.torchmodel import BaseModel

from .dataset import InferenceDataset

current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

class Slicer(BaseModel):
    def __init__(
        self,
        gpu_id: int = 0,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: Optional[List[int]] = None,
    ):
        super().__init__(
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.gpu_id = gpu_id
        if self.device_type == "gpu":
            self._set_gpu(self.gpu_id)
        elif self.device_type == "cpu":
            self.device = self._get_cpu()

        self.sph = {}  # Slice Parameters History
        self.recent_slice_key = None

        self._coordinates_type = "all_coordinates"
        self.tissue_geom = None

    def _set_wsi(self, wsi=None, tissue_geom=None, pass_wsi_object=False, **kwargs):
        """
        Sets the WSI object for the current instance by initializing a WSIManager.
    
        This method either creates a new WSIManager instance using the provided
        keyword arguments or directly assigns the given WSI object, based on the
        `pass_wsi_object` flag.
    
        Args:
            **kwargs: Additional arguments for initializing WSIManager, including:
                - `wsi_path` (str): The path to the WSI file.
                - `wsi_type` (str): The type of the WSI.
            wsi (optional): An existing WSI object to assign to the instance. Used only if `pass_wsi_object=True`.
            pass_wsi_object (bool, optional): Flag indicating whether to pass an existing WSI object (`True`) or
                initialize a new WSIManager (`False`). Defaults to `False`.
    
        Raises:
            ValueError: If required arguments (`wsi_path` and `wsi_type`) are missing when `pass_wsi_object=False`.
        """
        if not pass_wsi_object:
            if "wsi_path" not in kwargs or "wsi_type" not in kwargs:
                raise ValueError(
                    "`wsi_path` and `wsi_type` are required arguments in kwargs for WSIManager."
                )
    
            self.wsi = WSIManager(tissue_geom=tissue_geom, **kwargs).wsi
        else:
            self.wsi = wsi
            
        if tissue_geom:
            self._set_tissue_geom(tissue_geom=tissue_geom)
        else:
            self.tissue_geom = None
            self.tissue_geom_prepared = None

    def _set_tissue_geom(self, tissue_geom):
        self.tissue_geom = tissue_geom
        self.tissue_geom_prepared = prep_geom_for_query(tissue_geom)

    def _set_params(
        self,
        target_mpp: float,
        patch_size,
        overlap_size,
        context_size,
        slice_key=None,
        input_tuple: bool = False,
        show_progress: bool = False,
        set_coordinates: bool = True,
    ):
        slice_key = slice_key or str(self.wsi.stem)
        self.recent_slice_key = slice_key
        self.sph[slice_key] = {"params": {}}  # Slice Params
        params = self.sph[slice_key]["params"]
        params["target_mpp"] = target_mpp

        if input_tuple:
            params["patch_dims"] = patch_size
            params["overlap_dims"] = overlap_size
            params["context_dims"] = context_size
        else:
            params["patch_dims"] = (patch_size, patch_size)
            params["overlap_dims"] = (overlap_size, overlap_size)
            params["context_dims"] = (context_size, context_size)

        self._set_extraction_params(slice_key=slice_key)

        if set_coordinates:

            self.sph[slice_key]["all_coordinates"] = self.wsi._get_slice_wsi_coordinates(
                params
            )
    
            if self.tissue_geom is not None:
                self._filter_coordinates(slice_key, show_progress=show_progress)

    def _filter_coordinates(self, slice_key, show_progress=True):
        params = self.sph[slice_key]["params"]
        extraction_dims = params["extraction_dims"]
        factor1 = params["factor1"]
        coordinates = self.sph[slice_key]["all_coordinates"]

        box_height = extraction_dims[0] * factor1
        box_width = extraction_dims[1] * factor1

        tissue_contact_coordinates = []  # (coordinate, is_boundary)

        iterator = coordinates
        if show_progress:
            iterator = tqdm(coordinates, desc="Filtering coordinates")

        for (x, y), _ in iterator:
            box = get_box(x, y, box_height, box_width)
            if self.tissue_geom_prepared.intersects(box):
                if self.tissue_geom_prepared.contains(box):
                    tissue_contact_coordinates.append(((x, y), False))
                else:
                    tissue_contact_coordinates.append(((x, y), True))
                    #geom_region = self.tissue_geom.intersection(box).buffer(0)
                    #if geom_region.area == 0:
                        #tissue_contact_coordinates.append(((x, y), False))
                        #mask = np.zeros(extraction_dims, dtype=np.uint8)
                    #else:
                    #    tissue_contact_coordinates.append(((x, y), True))
                        #mask = self.get_numpy_mask_from_geom(
                        #    mask_dims=extraction_dims,
                        #    geom=geom_region,
                        #    origin=(x, y),
                        #    scale_factor=1 / factor1,
                        #)
                    #tissue_contact_coordinates.append(((x, y), mask))

        self.sph[slice_key]["tissue_contact_coordinates"] = tissue_contact_coordinates
        self._coordinates_type = "tissue_contact_coordinates"

    def _set_extraction_params(self, slice_key):
        """
        factor1 : factor to downsample from original_mpp to target_mpp
        factor2 : factor to downsample from original_mpp to downsample_mpp
        factor3 : factor to downsample from downsample_mpp to target_mpp
        """

        params = self.sph[slice_key]["params"]

        context_dims = params["context_dims"]
        overlap_dims = params["overlap_dims"]
        patch_dims = params["patch_dims"]

        params["shift_dims"] = (
            context_dims[0] + overlap_dims[0] // 2,
            context_dims[1] + overlap_dims[1] // 2,
        )

        params["shift_dims"] = (overlap_dims[0], overlap_dims[1])

        params["extraction_dims"] = (
            patch_dims[0] + 2 * context_dims[0],
            patch_dims[1] + 2 * context_dims[1],
        )

        params["stride_dims"] = (
            patch_dims[0] - (overlap_dims[0] + 2 * context_dims[0]),
            patch_dims[1] - (overlap_dims[1] + 2 * context_dims[1]),
        )

        params["factor1"] = self.wsi.factor_mpp(params["target_mpp"])
        params["level"] = self.wsi.get_level_for_downsample(params["factor1"])
        params["level_dims"] = self.wsi.level_mpp_dict[params["level"]]["dims"]
        params["mpp_at_level"] = self.wsi.level_mpp_dict[params["level"]]["mpp"]
        params["factor2"] = self.wsi.level_mpp_dict[params["level"]]["factor"]
        params["factor3"] = self.wsi.factor_mpp(
            target_mpp=params["target_mpp"], source_mpp=params["mpp_at_level"]
        )

        extraction_dims = params["extraction_dims"]
        factor3 = params["factor3"]
        params["extraction_dims_at_level"] = (
            self.round_to_nearest_even(extraction_dims[0] * factor3),
            self.round_to_nearest_even(extraction_dims[1] * factor3),
        )

        stride_dims = params["stride_dims"]
        params["stride_dims_at_level"] = (
            self.round_to_nearest_even(stride_dims[0] * factor3),
            self.round_to_nearest_even(stride_dims[1] * factor3),
        )

    def get_slice_region(self, coordinates, params):
        region = self.wsi.get_region_for_slicer(coordinates, params)
        return region

    def _set_slice_key(self, slice_key):
        """
        slice_key is intended to be the wsi name always, unless stated otherwise.
        """
        self.slice_key = slice_key

    def _worker_init_tiffslide(self, *args):
        self.wsi = TiffSlideWSI(
            wsi_path=self.wsi._wsi_path, tissue_geom=self.wsi.tissue_geom
        )

    def get_inference_dataloader(
        self,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        **kwargs,
    ):
        """
        Creates a PyTorch DataLoader for inference based on specified coordinate types.

        Parameters:
        - `batch_size` (int, optional): Number of samples per batch to load. Default is 2.
        - `shuffle` (bool, optional): Whether to shuffle the data at every epoch. Default is `False`.
        - `num_workers` (int, optional): Number of subprocesses to use for data loading. If set to 0, data will be loaded in the main process. Default is 0.
        - `**kwargs`: Additional arguments for `DataLoader`.

        Returns:
        - `DataLoader`: A PyTorch DataLoader for inference.

        Notes:
        - If the whole slide image type (`wsi.wsi_type`) is `"TiffSlide"` and `num_workers` is greater than 0, a custom worker initialization function `_worker_init_tiffslide` is used.
        """
        dataset = InferenceDataset(self)

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

    @staticmethod
    def round_to_nearest_even(x):
        return round(x / 2) * 2

    @staticmethod
    def get_numpy_mask_from_geom(
        geom, mask_dims: tuple, origin: tuple, scale_factor: float
    ):
        geom_dict = flatten_geom_collection(geom)
        if len(geom_dict) > 1:
            warnings.warn(
                f"Multiple geometries detected in tissue mask. Check: {', '.join(geom_dict.keys())}"
            )
        exterior, holes = [], []
        for polygon in geom_dict["Polygon"]:
            polygon_coordinates = get_polygon_coordinates_cpu(
                polygon, scale_factor=scale_factor, origin=origin
            )
            exterior.extend(polygon_coordinates[0])
            holes.extend(polygon_coordinates[1])
        mask = np.zeros(mask_dims, dtype=np.uint8)
        cv2.fillPoly(mask, exterior, 1)
        if len(holes) > 0:
            cv2.fillPoly(mask, holes, 0)
        return mask
