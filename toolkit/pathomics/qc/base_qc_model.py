from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from abc import ABC, abstractmethod

import torch

from toolkit.geometry.torch_tools import median_blur
from toolkit.system.storage.data_io_tools import h5, save_geojson
from toolkit.pathomics.torch.base_pathomics_model import BasePathomicsModel
from toolkit.geometry.shapely_tools import MultiPolygon, geom_to_geojson, loads

from toolkit.system.logging_tools import Logger

logger = Logger(name="qc").get_logger()


class BaseQCModel(BasePathomicsModel, ABC):
    def __init__(
        self,
        gpu_id: int = 0,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
    ):
        super().__init__(
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self._model_domain = "qc"

    def infer(
        self,
        show_progress=True,
        store_every_batch=False,
        replace_raw_predictions=False,
        **kwargs,
    ):
        self.timer.reset()
        self.timer.start_subtimer()

        if self.raw_predictions_path.exists() and not replace_raw_predictions:
            logger.warning(
                f"Inference already exists at {self.results_folder}, set replace_raw_predictions=True to replace the current file."
            )
            return

        self.model.eval()

        dataloader = self.get_inference_dataloader(**kwargs)

        autocast_dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        if show_progress:
            iterator = tqdm(dataloader, desc=f"Inference for {self._model_name}")
        else:
            iterator = dataloader

        with h5py.File(self.raw_predictions_path, "w") as h5file:
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type, dtype=autocast_dtype
            ):
                accumulated_predictions = []
                dataset = None

                for batch_idx, batch in enumerate(iterator):
                    batch = batch.to(self.device) - 0.5
                    pred_batch = self.model(batch)
                    pred_batch = torch.argmax(pred_batch, dim=1).float()
                    # pred_batch = median_blur(pred_batch.unsqueeze(1), 15).squeeze(1)
                    pred_array = pred_batch.to(torch.uint8).cpu().numpy()

                    if store_every_batch:
                        dataset = self._store_predictions(
                            h5file, "predictions", pred_array, dataset
                        )
                    else:
                        accumulated_predictions.append(pred_array)

                    del pred_batch, pred_array

                if not store_every_batch:
                    combined_predictions = np.concatenate(
                        accumulated_predictions, axis=0
                    )
                    self._store_predictions(h5file, "predictions", combined_predictions)

        self.custom_timer_metrics = {}
        self.custom_timer_metrics["wsi"] = self.slice_key
        kwargs_dict = {key: value for key, value in kwargs.items()}
        self.custom_timer_metrics.update(kwargs_dict)
        self.custom_timer_metrics["store_every_batch"] = store_every_batch
        self.custom_timer_metrics["autocast_dtype"] = str(autocast_dtype)
        self.custom_timer_metrics["model_name"] = self._model_name

        self.timer.stop_subtimer(process="inference")

        self.timer.set_custom_timer_metrics(self.custom_timer_metrics)
        self.timer.save_timer_logs()

    def process_predictions(
        self,
        show_progress=True,
        save_processed_predictions=True,
        replace_processed_predictions=False,
        replace_geojson=False,
        cmap_index=6,
    ):
        self.timer.reset()
        self.timer.start_subtimer()
        self.custom_timer_metrics = {}
        self.custom_timer_metrics["wsi"] = self.slice_key

        params = self.sph[self.slice_key]["params"]
        coordinates = self.sph[self.slice_key][self._coordinates_type]
        shift_dims = params["shift_dims"]
        scale_factor = params["factor1"]
        extraction_dims = params["extraction_dims"]
        box_width = extraction_dims[0] * scale_factor
        box_height = extraction_dims[1] * scale_factor

        shifted_dims = (shift_dims[0] * scale_factor, shift_dims[1] * scale_factor)
        processed_pred_dict = defaultdict(list)

        if (
            self.processed_predictions_path.exists()
            and not replace_processed_predictions
        ):
            logger.warning(
                f"h5 already exists at {self.processed_predictions_path}, set replace_processed_predictions=True to replace the current file."
            )
        else:
            with h5py.File(self.raw_predictions_path, "r") as h5file:
                predictions = h5file["predictions"]
                assert len(predictions) == len(coordinates)

                if show_progress:
                    iterator = tqdm(
                        zip(predictions, coordinates),
                        total=len(coordinates),
                        desc="Post processing",
                    )
                else:
                    iterator = zip(predictions, coordinates)

                for pred, ((x, y), boundary_status) in iterator:
                    pred = cv2.medianBlur(pred, ksize=self._med_blur_ksize)
                    if boundary_status:
                        mask = self._get_boundary_mask(
                            x, y, box_width, box_height, extraction_dims, scale_factor
                        )
                        pred *= mask
                    pred_sliced = pred[
                        shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]
                    ]
                    for predicted_class, value in self._class_map.items():
                        if predicted_class == "bg":  # Skip background
                            continue
                        class_mask = (pred_sliced == value).astype(np.uint8)
                        polys = self._get_prediction_geom(
                            x, y, class_mask, shifted_dims, scale_factor
                        )
                        if polys is not None:
                            processed_pred_dict[predicted_class].extend(polys)

                    if len(self._class_map) > 2:
                        pred_sliced[pred_sliced != 0] = 1
                        polys = self._get_prediction_geom(
                            x, y, pred_sliced, shifted_dims, scale_factor
                        )
                        if polys is not None:
                            processed_pred_dict["combined"].extend(polys)

            if save_processed_predictions:
                self._save_processed_predictions(
                    processed_pred_dict=processed_pred_dict,
                    show_progress=show_progress,
                )
            else:
                return processed_pred_dict

        if self.geojson_path.exists() and not replace_geojson:
            logger.warning(
                f"Geojson already exists at {self.geojson_path}, set replace_geojson=True to replace the current file."
            )
        else:
            self._save_geojson(show_progress=show_progress, cmap_index=cmap_index)

        self.timer.stop_subtimer(process="post processing")
        self.timer.set_custom_timer_metrics(self.custom_timer_metrics)
        self.timer.save_timer_logs()
