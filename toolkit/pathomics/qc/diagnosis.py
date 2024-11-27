from pathlib import Path

from typing import Optional, List

from .qc_models_v1.qc_models import (
    TissueModelV1,
    FocusModelV1,
    FoldsModelV1,
    PenModelV1,
    NodeDetectionV1,
)

from toolkit.system.logging_tools import Logger
from toolkit.geometry.shapely_tools import loads
from toolkit.system.storage.data_io_tools import h5
from toolkit.pathomics.torch.slicer import Slicer


class Diagnosis(Slicer):
    """
    Implements the Diagnosis class for executing quality control models on whole slide images (WSIs). This class is an extension of the Slicer class and integrates various models for tissue detection, focus analysis, fold detection, pen mark detection, and node detection.

    Args:
        gpu_id (int): ID of the GPU to be used for computation. Defaults to 0.
        device_type (str): Type of device to use, either "gpu" or "cpu". Defaults to "gpu".
        dataparallel (bool): Whether to enable data parallelism. Defaults to False.
        dataparallel_device_ids (List[int], optional): List of device IDs for data parallelism. Defaults to None.

    Methods:
        run_model_sequence:
            Executes a sequence of models for inference on a WSI. Prioritizes tissue detection models if required.

        list_available_models:
            Lists all available models by type and name.

        set_model:
            Instantiates a specified model with provided arguments and stores it as an attribute.

        _prioritize_tissue_model:
            Reorders the model run sequence to prioritize tissue detection models.

        _run_model:
            Executes inference for a specific model. Sets parameters, initializes the dataloader, and performs inference.
    """

    def __init__(
        self,
        default_tissue_detector: str = None,
        results_path: Path = None,
        gpu_id: int = 0,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: Optional[List[int]] = None,
    ):
        Slicer.__init__(
            self,
            results_path=results_path,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.logger = Logger(
            name="diagnosis", log_folder=f"logs/{results_path}"
        ).get_logger()

        self.logger.info(f"Initialised diagnosis object at: runs/{results_path}.")

        self.logger.debug(f"Results path set to: {self.results_path}.")

        self.default_tissue_detector = default_tissue_detector or "tissue_model_v1"
        self.logger.debug(
            f"Default tissue detector set to: {self.default_tissue_detector}."
        )

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
        model_run_sequence: list[str] = ["tissue_model_v1"],
        replace_model_results=False,
        replace_model=False,
        **kwargs,
    ) -> None:
        self.logger.info("Running model sequence.")
        if set(model_run_sequence) & set(self.tissue_detection_models):
            model_run_sequence = self._prioritize_tissue_model(model_run_sequence)
        else:
            model_run_sequence.insert(0, self.default_tissue_detector)

        for model_name in model_run_sequence:
            self.logger.info(f"Running model: {model_name}.")
            self._run_model(
                model_name=model_name,
                replace_model=replace_model,
                replace_model_results=replace_model_results,
                **kwargs,
            )

    def list_available_models(self):
        print("Listing available models.")
        for model_type, models in self.available_models.items():
            print(f"{model_type}:")
            for idx, model in enumerate(models):
                print(f"    {idx + 1}. {model}")

    def set_model(self, model_type: str, model_name: str, model_args: dict = None):
        self.logger.info(f"Setting model {model_name} of type {model_type}.")
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

    def _run_model(
        self,
        model_name,
        replace_model_results,
        replace_model,
        **kwargs,
    ):
        model = getattr(self, model_name, None)

        if model is None:
            self.list_available_models()
            self.logger.error(f"Model {model_name} not found in Diagnosis class.")
            raise ValueError(f"Model {model_name} not found in Diagnosis class.")

        model_results_folder = self.results_path / self.wsi.stem / Path("qc")
        model_results_folder.mkdir(parents=True, exist_ok=True)

        h5_path = model_results_folder / f"{model.model_name}.h5"
        # geojson_path = model_results_folder / f"{model.model_name}.geojson"

        if h5_path.exists():
            if replace_model_results:
                self.logger.info(f"replacing h5 results for {model_name} at {h5_path}.")
            else:
                self.logger.info(
                    f"h5 results for {model_name} already exist at {h5_path}, set replace_model_results=True for replacing the results."
                )
                return

        if not model.detects_tissue:
            if self.tissue_geom is None:
                tissue_geom_path = (
                    model_results_folder / f"{self.default_tissue_detector}.h5"
                )
                wkt_dict = h5.load_wkt_dict(tissue_geom_path)
                tissue_geom = loads(wkt_dict["combined"])
                self._set_tissue_geom(tissue_geom)
        
        if model.model is None or replace_model:
            model.load_model(replace_model=replace_model)

        self.set_params(
            target_mpp=model.mpp,
            patch_size=model.patch_size,
            overlap_size=model.overlap_size,
            context_size=model.context_size,
            slice_key=model.model_name,
        )
        self.set_slice_key(slice_key=model.model_name)
        dataloader = self.get_inference_dataloader(**kwargs)

        model.infer(
            dataloader=dataloader,
            model_results_folder=model_results_folder,
        )

    def _prioritize_tissue_model(self, model_run_sequence):
        self.logger.info("Prioritizing tissue detection model.")
        model_set = set(model_run_sequence)
        for model in self.tissue_detection_models:
            if model in model_set:
                model_run_sequence.remove(model)
                model_run_sequence.insert(0, model)
                break
        return model_run_sequence
