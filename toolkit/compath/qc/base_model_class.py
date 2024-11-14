import geojson
from pathlib import Path
from tqdm.auto import tqdm

import torch

from toolkit.load import h5, save_geojson
from toolkit.image_tools import get_rgb_colors, get_cmap

from toolkit.geometry.cv2_tools import get_contours, get_shapely_poly
from toolkit.geometry.torch_tools import median_blur
from toolkit.geometry.shapely_tools import MultiPolygon, loads, geom_to_geojson

from toolkit.vision.deep_learning.torchmodel import Model

class QCModel(Model):
    def __init__(
        self,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        Model.__init__(self, gpu_id, device_type, dataparallel, dataparallel_device_ids)

    def infer(
        self,
        dataloader,
        show_infer_progress=True,
        show_merge_preds_progress=True,
        show_process_merged_preds_progress=True,
    ):
        if self.model is None:
            raise ValueError("No model loaded, load a model first.")
        self.model.eval()

        self._set_inference_params(dataloader)
        self._pred_dicts = []

        if show_infer_progress:
            iterator = tqdm(self._dataloader, desc=f"Running {self.model_name} model")
        else:
            iterator = self._dataloader

        with torch.inference_mode():
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
            ):
                for batch, tissue_masks in iterator:
                    batch = batch.to(self.device) - 0.5
                    preds = self.model(batch)
                    preds = torch.argmax(preds, 1)

                    tissue_masks = tissue_masks.to(self.device)
                    preds *= tissue_masks
                    del tissue_masks

                    preds = preds.float().unsqueeze(1)
                    preds = median_blur(preds, 15)
                    preds = preds.squeeze(1).to(torch.uint8).to("cpu")

                    for pred in preds:
                        self._pred_dicts.append(self._process_pred(pred))

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        self._merge_preds(show_merge_preds_progress=show_merge_preds_progress)
        self._process_merged_preds(
            show_process_merged_preds_progress=show_process_merged_preds_progress
        )
        self._convert_to_geojson()

    def _set_inference_params(self, dataloader):
        self._dataloader = dataloader
        self._params = self._dataloader.dataset.params
        self._coordinates = [
            coordinate for coordinate, _ in self._dataloader.dataset.coordinates
        ]
        self.h5_path = Path(
            f"results/{self._dataloader.dataset.wsi_name.stem}/qc/h5/{self.model_name}.h5"
        )
        self.geojson_path = Path(
            f"results/{self._dataloader.dataset.wsi_name.stem}/qc/geojson/{self.model_name}.geojson"
        )

        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.geojson_path.parent.mkdir(parents=True, exist_ok=True)

    def _process_pred(self, pred):
        shift_dims = self._params["shift_dims"]
        pred_sliced = pred[
            shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]
        ]

        pred_dict = {}
        for key, value in self.class_map.items():
            if key == "bg":
                continue
            class_mask = (pred_sliced == value).to(torch.uint8).numpy()
            contours, hierarchy = get_contours(class_mask)
            pred_dict[key] = [contours, hierarchy]

        pred_sliced = pred_sliced.numpy()
        pred_sliced[pred_sliced != 0] = 1
        contours, hierarchy = get_contours(pred_sliced)
        pred_dict["combined"] = [contours, hierarchy]

        return pred_dict

    def _merge_preds(self, show_merge_preds_progress):
        assert len(self._coordinates) == len(
            self._pred_dicts
        ), "number of coordinates and predictions are unequal, can't merge back predictions together."

        self._processed_pred_dict = {}
        shift_dims = self._params["shift_dims"]
        scale_factor = self._params["factor1"]

        if show_merge_preds_progress:
            iterator = tqdm(
                zip(self._coordinates, self._pred_dicts),
                total=len(self._pred_dicts),
                desc="Merging predictions",
            )
        else:
            iterator = zip(self._coordinates, self._pred_dicts)

        for (x, y), pred_dict in iterator:
            for key, (contours, hierarchy) in pred_dict.items():
                if len(contours) == 0:
                    continue
                polys = get_shapely_poly(
                    contours,
                    hierarchy,
                    scale_factor=scale_factor,
                    shift_x=x + int(shift_dims[0] * scale_factor),
                    shift_y=y + int(shift_dims[1] * scale_factor),
                )
                if key in self._processed_pred_dict:
                    self._processed_pred_dict[key].extend(polys)
                else:
                    self._processed_pred_dict[key] = []
                    self._processed_pred_dict[key].extend(polys)

    def _process_merged_preds(self, show_process_merged_preds_progress):
        wkt_dict = {}

        if show_process_merged_preds_progress:
            iterator = tqdm(
                self._processed_pred_dict.items(), desc="Processing predictions"
            )
        else:
            iterator = self._processed_pred_dict.items()

        for key, value in iterator:
            if show_process_merged_preds_progress:
                iterator.set_description(f"Processing {key.capitalize()}")
            wkt_dict[key] = MultiPolygon(value).buffer(0).wkt

        h5.save_wkt_dict(wkt_dict, self.h5_path)

    def _convert_to_geojson(self):
        wkt_dict = h5.load_wkt_dict(self.h5_path)
        colors = get_rgb_colors(len(wkt_dict), cmap=get_cmap(6))
        geojson_features = []

        for idx, dict_item in enumerate(wkt_dict.items()):
            key, value = dict_item
            geojson_feature = geom_to_geojson(loads(value))
            geojson_feature["properties"] = {
                "objectType": "annotation",
                "name": key,
                "color": colors[idx],
            }
            geojson_features.append(geojson_feature)
        geojson_feature_collection = geojson.FeatureCollection(geojson_features)
        save_geojson(
            geojson_feature_collection,
            self.geojson_path,
        )
