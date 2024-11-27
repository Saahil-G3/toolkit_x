import h5py
import torch
import geojson
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional
from datetime import datetime
from collections import defaultdict

from toolkit.system.logging_tools import Timer
from toolkit.geometry.torch_tools import median_blur
from toolkit.pathomics.dataloading.torch.slicer import Slicer
from toolkit.vision.image_tools import get_cmap, get_rgb_colors
from toolkit.system.storage.data_io_tools import h5, save_geojson
from toolkit.geometry.cv2_tools import get_contours, get_shapely_poly
from toolkit.geometry.shapely_tools import MultiPolygon, geom_to_geojson, loads

from toolkit.system.logging_tools import Logger

logger = Logger(name="base_qc_model").get_logger()


class _BaseQCModel(Slicer):
    def __init__(
        self,
        gpu_id: int = 0,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: Optional[list[int]] = None,
    ):
        Slicer.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        self.timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

    def set_wsi(self, result_folder=None, **kwargs1):

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

    def set_base_folder(self, base_folder: Path = None) -> None:

        self.base_folder = Path(base_folder) if base_folder else Path(self.timestamp)
        self.timer = Timer(
            timer_name="qc_model", logs_folder=self.base_folder, save_logs=True
        )

    def _set_result_paths(self) -> None:

        self.results_folder = (
            self.base_folder / self.slice_key / "qc" / self._model_name
        )
        self.results_folder.mkdir(exist_ok=True, parents=True)
        self.h5_path = self.results_folder / f"{self._model_name}.h5"
        self.geojson_path = self.results_folder / f"{self._model_name}.geojson"

        self.custom_timer_metrics = {}
        self.custom_timer_metrics["wsi"] = self.slice_key

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

    def infer(
        self,
        store_every_batch: bool = False,
        show_infer_progress: bool = True,
        replace_predictions=False,
        **kwargs,
    ):
        self.predictions_path = (
            self.results_folder
            / f"predictions_ps:{self._patch_size}_os:{self._overlap_size}_cs:{self._context_size}.h5"
        )
        if self.predictions_path.exists() or not replace_predictions:
            logger.warning(
                f"Inference already exists at {self.results_folder}, set replace_predictions=True to replace the current file.."
            )
            return

        dataloader = self.get_inference_dataloader(**kwargs)
        self.timer.start()
        self.model.eval()

        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        accumulated_predictions = []

        iterator = (
            tqdm(dataloader, desc=f"Inference for {self._model_name}")
            if show_infer_progress
            else dataloader
        )

        with h5py.File(self.predictions_path, "w") as h5file:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=autocast_dtype
            ):
                dataset = None  # Initialize dataset handle

                for batch_idx, batch in enumerate(iterator):
                    batch = batch.to(self.device) - 0.5
                    pred_batch = self.model(batch)
                    pred_batch = torch.argmax(pred_batch, dim=1).float().unsqueeze(1)
                    pred_batch = median_blur(pred_batch, 15).squeeze(1)
                    pred_array = pred_batch.to(torch.uint8).cpu().numpy()

                    if store_every_batch:
                        dataset = self._store_predictions(
                            h5file, "predictions", pred_array, dataset
                        )
                    else:
                        accumulated_predictions.append(pred_array)

                if not store_every_batch:
                    combined_predictions = np.concatenate(
                        accumulated_predictions, axis=0
                    )
                    self._store_predictions(h5file, "predictions", combined_predictions)

        kwargs_dict = {key: value for key, value in kwargs.items()}
        self.custom_timer_metrics.update(kwargs_dict)
        self.custom_timer_metrics["store_every_batch"] = store_every_batch
        self.custom_timer_metrics["autocast_dtype"] = str(autocast_dtype)
        self.custom_timer_metrics["model_name"] = self._model_name
        self.timer.set_custom_timer_metrics(self.custom_timer_metrics)
        self.timer.stop()

    def process_predictions(
        self,
        save_h5=True,
        save_geojson=True,
        show_save_h5_progress=True,
        show_save_geojson_progress=False,
        show_process_predictions_progress=True,
        replace_geojson=False,
        replace_h5=False,
    ):

        params = self.sph[self.slice_key]["params"]
        coordinates = self.sph[self.slice_key][self._coordinates_type]
        shift_dims = params["shift_dims"]
        scale_factor = params["factor1"]
        extraction_dims = params["extraction_dims"]
        box_width = extraction_dims[0] * scale_factor
        box_height = extraction_dims[1] * scale_factor

        shifted_dims = (shift_dims[0] * scale_factor, shift_dims[1] * scale_factor)
        processed_pred_dict = defaultdict(list)

        if self.h5_path.exists() or not replace_h5:
            logger.warning(
                f"h5 already exists at {self.h5_path}, set replace_h5=True to replace the current file."
            )
        else:
            with h5py.File(self.predictions_path, "r") as h5file:
                predictions = h5file["predictions"]
                assert len(predictions) == len(coordinates)
                if show_process_predictions_progress:
                    iterator = tqdm(
                        zip(predictions, coordinates),
                        total=len(coordinates),
                        desc="Post processing",
                    )
                else:
                    iterator = zip(predictions, coordinates)
                for pred, ((x, y), boundary_status) in iterator:
                    if boundary_status:
                        mask = self._get_boundary_mask(
                            x, y, box_width, box_height, extraction_dims, scale_factor
                        )
                        pred *= mask

                    pred_sliced = pred[
                        shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]
                    ]
                    polys, predicted_class = self._get_prediction_geom(
                        x, y, pred_sliced, shifted_dims, scale_factor
                    )

                    if polys is not None:
                        processed_pred_dict[predicted_class].extend(polys)

            if save_h5:
                self._save_h5(
                    processed_pred_dict=processed_pred_dict,
                    show_save_h5_progress=show_save_h5_progress,
                )

            else:
                return processed_pred_dict

        if self.geojson_path.exists() or not replace_geojson:
            logger.warning(
                f"Geojson already exists at {self.geojson_path}, set replace_geojson=True to replace the current file."
            )
        elif save_geojson or replace_geojson:
            self._save_geojson(show_save_geojson_progress=show_save_geojson_progress)

    def _save_geojson(self, wkt_dict_path=None, show_save_geojson_progress=False):
        if wkt_dict_path:
            wkt_dict = h5.load_wkt_dict(wkt_dict_path)
        else:
            wkt_dict = h5.load_wkt_dict(self.h5_path)
        colors = get_rgb_colors(len(wkt_dict) + 1, cmap=get_cmap(6))
        geojson_features = []

        if show_save_geojson_progress:
            iterator = tqdm(enumerate(wkt_dict.items()), desc="Saving geojson")
        else:
            iterator = enumerate(wkt_dict.items())

        for idx, dict_item in iterator:
            key, value = dict_item
            geojson_feature = geom_to_geojson(loads(value))
            geojson_feature["properties"] = {
                "objectType": "annotation",
                "name": key,
                "color": colors[idx + 1],
            }
            geojson_features.append(geojson_feature)
        geojson_feature_collection = geojson.FeatureCollection(geojson_features)
        save_geojson(
            geojson_feature_collection,
            self.geojson_path,
        )

    def _save_h5(self, processed_pred_dict, show_save_h5_progress=True):
        wkt_dict = {}

        if show_save_h5_progress:
            iterator = tqdm(processed_pred_dict.items(), desc="Processing h5")
        else:
            iterator = self._processed_pred_dict.items()

        for key, value in iterator:
            if show_save_h5_progress:
                iterator.set_description(f"Processing h5 for {key.capitalize()}")
            wkt_dict[key] = MultiPolygon(value).buffer(0).wkt

        h5.save_wkt_dict(wkt_dict, self.h5_path)

    def _get_prediction_geom(self, x, y, pred_sliced, shifted_dims, scale_factor):
        for predicted_class, value in self._class_map.items():
            if predicted_class == "bg":  # Skip background
                continue
            class_mask = (pred_sliced == value).astype(np.uint8)
            contours, hierarchy = get_contours(class_mask)
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

            return polys, predicted_class

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
