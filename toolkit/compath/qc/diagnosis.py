from pathlib import Path
from typing import Optional, List

from .qc_models_v1.qc_models import TissueModelV1, FocusModelV1, FoldsModelV1, PenModelV1, NodeDetectionV1

from toolkit.compath.dataloading.slicer import Slicer
from toolkit.geometry.shapely_tools import loads
from toolkit.system.storage.data_io_tools import h5


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
        
        self._default_tissue_detector = "tissue_model_v1"
        self.tissue_detection_models = ["tissue_model_v1", "node_detection_v1"]
        self.model_types = ["qc_models_v1"] 
        self.available_models = {
            "qc_models_v1": {
                "tissue_model_v1": TissueModelV1,
                "focus_model_v1": FocusModelV1,
                "folds_model_v1": FoldsModelV1,
                "pen_model_v1": PenModelV1,
                "node_detection_v1": NodeDetectionV1,
            }
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
        for model_type, models in self.available_models.items():
            print(f"{model_type}:")
            print("\n".join(f"    {idx + 1}. {model}" for idx, model in enumerate(models)))
            print()

    def set_model(self, model_type: str, model_name: str, model_args: dict = None):
        """
        Sets the specified model by instantiating it with the provided arguments and storing it as an attribute of the class instance.
    
        Args:
            model_type (str): The type of model to set. This determines which set of available models to choose from (e.g., "qc_models_v1").
            model_name (str): The name of the specific model to instantiate (e.g., "node_detection_v1").
            model_args (dict, optional): A dictionary containing the arguments to be passed to the model's constructor. Defaults to None.
    
        Raises:
            KeyError: If the provided `model_type` or `model_name` does not exist in the `available_models` dictionary.
    
        Example:
            # Setting the node detection model with specific arguments
            node_detection_v1_args = {
                "gpu_id": 0,
                "patch_size": 256,
                "device_type": "gpu",
                "dataparallel": None
            }
            set_model("qc_models_v1", "node_detection_v1", node_detection_v1_args)
        """
        available_types = ", ".join(self.available_models.keys())
        available_names = (
            ", ".join(self.available_models[model_type].keys())
            if model_type in self.available_models
            else "N/A"
        )
    
        if (
            model_type not in self.available_models
            or model_name not in self.available_models.get(model_type, {})
        ):
            raise ValueError(
                f"Invalid model_type or model_name.\n"
                f"Available model types: {available_types}\n"
                f"Available models for type '{model_type}': {available_names}\n"
                f"Requested model: {model_name}"
            )

        if model_args is None:
            model_args = {
                "gpu_id": 0,
                "patch_size": 256,
                "device_type": "gpu",
                "dataparallel": None,
                "dataparallel_device_ids": None,
            }
    
        model_class = self.available_models[model_type][model_name]
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
