from pathlib import Path
from typing import Optional, List

from .qc_models_v1 import _TissueModelV1, _FocusModelV1, _FoldsModelV1, _PenModelV1

from toolkit.compath.dataloading.slicer import Slicer
from toolkit.geometry.shapely_tools import loads
from toolkit.system.storage.load import h5


class Diagnosis(Slicer):
    def __init__(
        self,
        gpu_id: int =0,
        device_type: str ="gpu",
        dataparallel: bool =False,
        data_loading_mode: str ="cpu",
        dataparallel_device_ids: Optional[List[int]]=None,
    ):
        Slicer.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            data_loading_mode=data_loading_mode,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self._set_models()
        self._default_tissue_detector = "tissue_model_v1"
        self.tissue_detection_models = ["tissue_model_v1"]
        self.available_models = {
            "qc_models_v1": [
                "tissue_model_v1",
                "focus_model_v1",
                "folds_model_v1",
                "pen_model_v1",
            ]
        }

        self.model_name_mapping = {
            "tissue_model_v1": _TissueModelV1,
            "focus_model_v1": _FocusModelV1,
            "folds_model_v1": _FoldsModelV1,
            "pen_model_v1": _PenModelV1,
        }

    def run_model_sequence(
        self,
        wsi_path: str,
        wsi_type: str,
        model_run_sequence: list[str] = ["tissue_model_v1"],
        **kwargs,
    )-> None:
        self.set_wsi(wsi_path=wsi_path, wsi_type=wsi_type)
        self.tissue_path = Path(
            f"results/{self.wsi._wsi_path.stem}/qc/h5/{self._default_tissue_detector}.h5"
        )

        if not self.tissue_path.exists():
            if set(model_run_sequence) & set(self.tissue_detection_models):
                model_run_sequence = self._prioritize_tissue_model(model_run_sequence)
            else:
                model_run_sequence.insert(0, self._default_tissue_detector)

        for model_name in model_run_sequence:
            if Path(
                f"results/{self.wsi._wsi_path.stem}/qc/h5/{model_name}.h5"
            ).exists():
                print(
                    f"h5 results for {model_name} already exists skipping to next model."
                )
                continue
            self._run_model(model_name=model_name, **kwargs)

    def list_available_models(self):
        print("Models available:")
        for key, value in self.available_models.items():
            print(f"{key}:")
            for idx, available_model in enumerate(value):
                print(f"    {idx+1}. {available_model}")
            print("\n")

    def set_model(self, model_name: str, model_args: dict = None):
        model_class = self.model_mapping[model_name]
        model_instance = model_class(**model_args)
        setattr(self, f"{model_name}", model_instance)

    def _prioritize_tissue_model(self, model_run_sequence):
        model_set = set(model_run_sequence)
        for model in self.tissue_detection_models:
            if model in model_set:
                model_run_sequence.remove(model)
                model_run_sequence.insert(0, model)
                break
        return model_run_sequence

    def _run_model(
        self,
        model_name,
        replace_model=False,
        **kwargs,
    ):
        if model_name not in self.tissue_detection_models:
            tissue_wkt_dict = h5.load_wkt_dict(self.tissue_path)
            tissue_geom = loads(tissue_wkt_dict["combined"])
            self.set_tissue_geom(tissue_geom)

        model = getattr(self, model_name, None)
        if model is None:
            self.list_available_models()
            raise ValueError(f"Model {model_name} not found in Diagnosis class.")
        model.load_model(replace=replace_model)

        self.set_params(
            target_mpp=model.mpp,
            patch_size=model.patch_size,
            overlap_size=model.overlap_size,
            context_size=model.context_size,
            slice_key=model.model_name,
        )
        self.set_slice_key(slice_key=model.model_name)
        dataloader = self.get_inference_dataloader(**kwargs)

        model.infer(dataloader)
