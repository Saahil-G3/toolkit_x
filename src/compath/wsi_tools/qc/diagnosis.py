import geojson
from cv2 import medianBlur
from tqdm.auto import tqdm
from torch import load as torch_load
from torch import autocast as torch_autocast
from torch import float32 as torch_float32
from torch import float16 as torch_float16
from torch import argmax as torch_argmax

from numpy import uint8 as np_uint8
import segmentation_models_pytorch as smp
from pathlib import Path

from geometry.cv2_tools import get_contours, get_shapely_poly
from geometry.shapely_tools import MultiPolygon, loads, geom_to_geojson
from image_tools import get_rgb_colors, get_cmap
from compath.wsi_tools.pytorch_slicer import Slicer
from load import H5, save_geojson
h5 = H5()

class Diagnosis(Slicer):
    def __init__(self, wsi, device):
        Slicer.__init__(self, wsi)
        self._set_metadata()
        self.device = device
        
        self.qc_folder = Path(f"{wsi._wsi_path.parent}/qc")
        self.qc_folder.mkdir(exist_ok=True)
        #wkt_dict_path = Path(f"{qc_folder}/{model_type}.h5")

    def _set_metadata(self):
        self.metadata = {}

        # Metadata for tissue model
        self.metadata["tissue"] = {
            "path": "weights/tissue.pt",
            "class_map": {"bg": 0, "adipose": 1, "non_adipose": 2},
            "mpp": 4,
            "model_config": {
                "encoder_name": "resnet18",
                "in_channels": 3,
                "classes": 3,
            },
        }

        # Metadata for folds model
        self.metadata["folds"] = {
            "path": "weights/folds.pt",
            "class_map": {"bg": 0, "fold": 1},
            "mpp": 2,
            "model_config": {
                "encoder_name": "resnet18",
                "in_channels": 3,
                "classes": 2,
            },
        }

        # Metadata for focus model
        self.metadata["focus"] = {
            "path": "weights/blur.pt",
            "class_map": {
                "bg": 0,
                "level_1": 1,
                "level_2": 2,
                "level_3": 3,
                "level_4": 4,
            },
            "mpp": 2,
            "model_config": {
                "encoder_name": "resnet18",
                "in_channels": 3,
                "classes": 5,
            },
        }

        # Metadata for pen model
        self.metadata["pen"] = {
            "path": "weights/pen.pt",
            "class_map": {"bg": 0, "pen_mark": 1},
            "mpp": 16,
            "model_config": {
                "encoder_name": "resnet34",
                "in_channels": 3,
                "classes": 2,
            },
        }

    def run_tissue_model(self, patch_size = 1024, save_geojson_mask=True):
        model_type = 'tissue'
        self.load_model(model_type)
        overlap_size = int(patch_size * (0.0625))
        context_size = overlap_size
        
        self.set_params(self.metadata[model_type]['mpp'], patch_size, overlap_size, context_size, slice_key=model_type)
        self.set_slice_key(slice_key=model_type)
        
        self.metadata[model_type]['dataloader'] = self.get_dataloader(batch_size=2)
        
        self.sph[model_type]['pred_dicts'] = self.run_inference(model_type)
        
        assert len(self.sph[model_type]['coordinates']) == len(
            self.sph[model_type]['pred_dicts']
        ), "number of coordintes and predictions are unequal, cant merge back predictions together."
        
        processed_pred_dict = self._post_process_pred_dicts(model_type)
        
        wkt_dict = {}
        for key, value in processed_pred_dict.items():
            wkt_dict[key] = MultiPolygon(processed_pred_dict[key]).buffer(0).wkt
            
        h5.save_wkt_dict(wkt_dict, f"{self.qc_folder}/{model_type}.h5")

        if save_geojson_mask:
            colors = get_rgb_colors(len(wkt_dict), cmap=get_cmap(6))
            geojson_features= []
            for idx, dict_item in enumerate(wkt_dict.items()):
                key, value = dict_item
                geojson_feature = geom_to_geojson(loads(value))
                geojson_feature['properties'] = {'objectType': 'annotation','name':key , 'color': colors[idx]}
                geojson_features.append(geojson_feature)
            geojson_feature_collection = geojson.FeatureCollection(geojson_features)
            save_geojson(geojson_feature_collection, f"{self.qc_folder}/{model_type}.geojson")
    
    def _load_and_store_model(self, model_type, config):
        model = smp.UnetPlusPlus(
            encoder_name=config["encoder_name"],
            encoder_weights="imagenet",
            in_channels=config["in_channels"],
            classes=config["classes"],
        )
        model.load_state_dict(
            torch_load(self.metadata[model_type]["path"], map_location=self.device, weights_only=True)
        )
        model = model.eval().to(self.device)
        self.metadata[model_type]["model"] = model
        
    def load_model(self, model_type):
        if model_type == "all":
            for key, data in self.metadata.items():
                if "model" not in data or data["model"] is None:
                    self._load_and_store_model(key, data["model_config"])
        else:
            if model_type in self.metadata:
                if "model" not in self.metadata[model_type] or self.metadata[model_type]["model"] is None:
                    self._load_and_store_model(model_type, self.metadata[model_type]["model_config"])
                else:
                    print(f"Model '{model_type}' is already loaded.")
            else:
                raise KeyError(f"Model '{model_type}' not found in metadata.")

    def run_inference(self, model_type):
        pred_dicts = []
        with torch_autocast(device_type=self.device.type, dtype=torch_float16 if self.device.type == 'cuda' else torch_float32):
            for batch in tqdm(self.metadata[model_type]['dataloader'], desc=f"running {model_type} model"):
                batch = batch.to(self.device)
                batch = (batch / 255) - 0.5
                preds_batch = self.metadata[model_type]['model'](batch)
                preds_batch = torch_argmax(preds_batch, 1)
                preds_batch = preds_batch.cpu().numpy().astype(np_uint8)
    
                for pred_mask in preds_batch:
                    pred_dict = self._process_pred_mask(
                        pred_mask=pred_mask,
                        model_type=model_type,
                        )
                    pred_dicts.append(pred_dict)
                del preds_batch
        return pred_dicts
        
    def _process_pred_mask(self, pred_mask, model_type):
        shift_dims = self.sph[model_type]['params']['shift_dims']
        pred_dict = {}
        med_blur = medianBlur(pred_mask, 15)
    
        for key, value in self.metadata[model_type]['class_map'].items():
            if key == "bg":
                continue
            class_mask = np_uint8(med_blur == value).copy()
            contours, hierarchy = get_contours(
                class_mask[shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]]
            )
            pred_dict[key] = [contours, hierarchy]
    
        med_blur[med_blur != 0] = 1
        contours, hierarchy = get_contours(
            med_blur[shift_dims[0] : -shift_dims[0], shift_dims[1] : -shift_dims[1]]
        )
        pred_dict["combined"] = [contours, hierarchy]
    
        return pred_dict

    def _post_process_pred_dicts(self, model_type):
        shift_dims = self.sph[model_type]["params"]["shift_dims"]
        scale_factor = self.sph[model_type]["params"]["factor1"]
        processed_pred_dict = {}
    
        for pred_dict, coordinate in zip(
            self.sph[model_type]["pred_dicts"], self.sph[model_type]["coordinates"]
        ):
            x, y = coordinate
            for key, value in pred_dict.items():
                contours, hierarchy = value
                if len(contours) == 0:
                    continue
                polys = get_shapely_poly(
                    contours,
                    hierarchy,
                    scale_factor=scale_factor,
                    shift_x=x + int(shift_dims[0] * scale_factor),
                    shift_y=y + int(shift_dims[1] * scale_factor),
                )
                if key in processed_pred_dict:
                    processed_pred_dict[key].extend(polys)
                else:
                    processed_pred_dict[key] = []
                    processed_pred_dict[key].extend(polys)
    
        return processed_pred_dict
