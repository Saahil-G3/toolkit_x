import h5py
import geojson
import numpy as np
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod

from tqdm.auto import tqdm

from toolkit.system.logging_tools import Timer
from toolkit.pathomics.torch.slicer import Slicer
from toolkit.vision.colors import get_rgb_colors, get_cmap
from toolkit.system.storage.data_io_tools import h5, save_geojson
from toolkit.geometry.cv2_tools import get_contours, get_shapely_poly

from toolkit.geometry.shapely_tools import MultiPolygon
from toolkit.geometry.shapely_tools import get_box, geom_to_geojson, loads


class BasePathomicsModel(Slicer, ABC):
    """
    Every inititation of a class that is inheriting _BasePathomicsModel must have these 2 methods defined:

    1. _set_model_specific_params, It initializes model-specific parameters.

        Example of implementation:
        ```python
        def _set_model_specific_params(self) -> None:
            self._model_domain = "germinal_center" #Represents the domain or subject of the model, could be defined out of this function in class __init__ as well.
            self._detects_tissue = False #Whether the model is a tissue detector or not.
            self._model_name = "gc_mpp1"
            self._class_map = {"bg": 0, "germinal_center": 1}
            self._mpp = 1
            self._probability_threshold = 0.2 #Only if threshold is tuned and going to be implemented in the inference logic.
        ```
    2. _set_model_class, It initializes the model class and architecture.
        Example of implementation:
        ```python
        def _set_model_class(self):
            self._state_dict_path = Path(f"weights/my_model/")
            self._model_class = "smp"
            #if self._model_class is "smp" the following attributes must also be present.
            self._architecture = "UnetPlusPlus"
            self._encoder_name = "resnet34"
            self._encoder_weights = "imagenet"
            self._in_channels = 3
            self._classes = 2

            #if self._model_class is "custom", then self._architecture must be the Model configuration where the state_dict will be mapped.
        ```
    """

    def __init__(
        self,
        gpu_id: int = 0,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
        state_dict_path = None,
    ):
        super().__init__(
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.state_dict_path = state_dict_path
        
        self._set_model_specific_params()
        self._set_model_class()

    def _get_timestamp(self):
        return datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    def set_base_folder(self, base_folder: Path = None) -> None:

        base_folder = base_folder if base_folder else self._get_timestamp()
        
        self.base_folder = Path(f"runs/{base_folder}")
        self.logs_folder = Path(f"logs/{base_folder}")
        self.logs_folder.mkdir(parents=True, exist_ok=True)
        self.base_folder.mkdir(parents=True, exist_ok=True)

        self.timer = Timer(
            timer_name=self._model_name, logs_folder=self.logs_folder
        )

    def set_wsi(
        self,
        patch_size=1024,
        overlap_size=None,
        context_size=None,
        **kwargs1,
    ):

        self._patch_size = patch_size
        self._overlap_size = overlap_size or int(self._patch_size * 0.0625)
        self._context_size = context_size

        self._set_wsi(**kwargs1)
        slice_key = str(self.wsi.stem)

        self._set_params(
            target_mpp=self._mpp,
            patch_size=self._patch_size,
            overlap_size=self._overlap_size,
            context_size=self._context_size,
            slice_key=slice_key,
        )

        self._set_slice_key(slice_key=slice_key)
        self._set_result_paths()

    def _set_result_paths(self) -> None:
        self.results_folder = (
            self.base_folder / self.slice_key / self._model_domain / self._model_name
        )
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.raw_predictions_path = self.results_folder / f"raw_predictions.h5"
        self.processed_predictions_path = (
            self.results_folder / f"processed_predictions_wkt.h5"
        )
        self.geojson_path = self.results_folder / f"{self._model_name}.geojson"

    def _store_predictions(
        self,
        h5file: h5py.File,
        dataset_name: str,
        predictions: np.ndarray,
        dataset: Optional[h5py.Dataset] = None,
        dtype="uint8",
        compression="gzip",
    ) -> h5py.Dataset:
        """
        Store predictions into an HDF5 file, either initializing a dataset or appending to it.

        Args:
            h5file (h5py.File): Open HDF5 file handle.
            dataset_name (str): Name of the dataset to store predictions in.
            predictions (np.ndarray): Array of predictions to store.
            dataset (Optional[h5py.Dataset]): Existing HDF5 dataset to append to, if any.

        Returns:
            h5py.Dataset: Updated HDF5 dataset handle.
        """
        if dataset is None:
            dataset = h5file.create_dataset(
                dataset_name,
                data=predictions,
                maxshape=(None, *predictions.shape[1:]),
                chunks=True,
                dtype=dtype,
                compression=compression,
                # compression_opts=1,
            )
        else:
            current_size = dataset.shape[0]
            new_size = current_size + predictions.shape[0]
            dataset.resize(new_size, axis=0)
            dataset[current_size:new_size] = predictions

        return dataset

    def _save_geojson(
        self,
        wkt_dict_path=None,
        show_progress=True,
        replace_geojson=False,
        cmap_index=6,
    ):

        if wkt_dict_path:
            wkt_dict = h5.load_wkt_dict(wkt_dict_path)
        else:
            wkt_dict = h5.load_wkt_dict(self.processed_predictions_path)
        colors = get_rgb_colors(len(wkt_dict) + 1, cmap=get_cmap(cmap_index))
        geojson_features = []

        if show_progress:
            iterator = tqdm(
                enumerate(wkt_dict.items()), total=len(wkt_dict), desc="Saving geojson"
            )
        else:
            iterator = enumerate(wkt_dict.items())

        for idx, dict_item in iterator:
            key, value = dict_item
            geojson_feature = geom_to_geojson(loads(value))
            geojson_feature["properties"] = {
                "objectType": "annotation",
                "name": f"{key} ({self._model_name})",
                "color": colors[idx + 1],
            }
            geojson_features.append(geojson_feature)
        geojson_feature_collection = geojson.FeatureCollection(geojson_features)
        save_geojson(
            geojson_feature_collection,
            self.geojson_path,
        )

    def _save_processed_predictions(self, processed_pred_dict, show_progress=True):
        wkt_dict = {}

        if show_progress:
            iterator = tqdm(processed_pred_dict.items(), desc="Processing predictions")
        else:
            iterator = processed_pred_dict.items()

        for key, value in iterator:
            if show_progress:
                iterator.set_description(
                    f"Processing prediction for {key.capitalize()}"
                )
            wkt_dict[key] = MultiPolygon(value).buffer(0).wkt

        h5.save_wkt_dict(wkt_dict, self.processed_predictions_path)

    def _get_boundary_mask(
        self, x, y, box_width, box_height, extraction_dims, scale_factor
    ):
        box = get_box(x, y, box_width, box_height)
        geom_region = self.tissue_geom.intersection(box).buffer(0)
        if geom_region.area == 0:
            mask = np.zeros(extraction_dims, dtype=np.uint8)
        else:
            mask = self.get_numpy_mask_from_geom(
                mask_dims=extraction_dims,
                geom=geom_region,
                origin=(x, y),
                scale_factor=1 / scale_factor,
            )

        return mask

    def _get_prediction_geom(self, x, y, mask, shifted_dims, scale_factor):
        contours, hierarchy = get_contours(mask)
        if contours:  # Process only if contours exist
            polys = get_shapely_poly(
                contours,
                hierarchy,
                scale_factor=scale_factor,
                shift_x=x + int(shifted_dims[0]),
                shift_y=y + int(shifted_dims[1]),
            )
        else:
            polys = None

        return polys
        
    def get_class_geom(self, class_name):
        wkt_dict = h5.load_wkt_dict(self.processed_predictions_path)
        mgeom = loads(wkt_dict[class_name])
        return mgeom

    @abstractmethod
    def _set_model_specific_params(self):
        pass

    @abstractmethod
    def _set_model_class(self):
        pass
