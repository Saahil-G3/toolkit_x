import logging
# Set up logging
logging.basicConfig(
    filename='qc_logs.txt',  # File where logs will be saved
    level=logging.INFO,          # Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
)

from pathlib import Path

from toolkit.compath.dataloading.slicer import Slicer

from .qc_models_v1 import TissueModelV1, FocusModelV1, FoldsModelV1, PenModelV1

from toolkit.geometry.shapely_tools import loads
from toolkit.system.storage.load import h5

class Diagnosis(Slicer):
    def __init__(
        self,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        data_loading_mode="cpu",
        dataparallel_device_ids=None,
    ):
        Slicer.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            data_loading_mode=data_loading_mode,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        self._qc_model_gpu_args = {
            "gpu_id": gpu_id,
            "device_type": device_type,
            "dataparallel": dataparallel,
            "dataparallel_device_ids": dataparallel_device_ids,
        }
        self._set_models()
        self.slicer = slicer = Slicer()
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

    def run_model_sequence(
        self,
        wsi_path: str,
        wsi_type: str,
        model_run_sequence: list = ["tissue_model_v1"],
        **kwargs,
    ):
        self.set_wsi(wsi_path=wsi_path, wsi_type=wsi_type)
        logging.info(f"Initialised {wsi_type} object for {wsi_path.name}")
        self.tissue_path = Path(
            f"results/{self.wsi._wsi_path.stem}/qc/h5/{self._default_tissue_detector}.h5"
        )
        
        if not self.tissue_path.exists():
            if set(model_run_sequence) & set(self.tissue_detection_models):
                model_run_sequence = self._prioritize_tissue_model(model_run_sequence)
            else:
                model_run_sequence.insert(0, self._default_tissue_detector)
                logging.info(
                    f"No tissue detector in the sequence, added {self._default_tissue_detector}"
                )
                
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

    def _set_models(self):
        self.tissue_model_v1 = TissueModelV1(**self._qc_model_gpu_args)
        self.focus_model_v1 = FocusModelV1(**self._qc_model_gpu_args)
        self.folds_model_v1 = FoldsModelV1(**self._qc_model_gpu_args)
        self.pen_model_v1 = PenModelV1(**self._qc_model_gpu_args)

    def _prioritize_tissue_model(self, model_run_sequence):
        model_set = set(model_run_sequence)
        for model in self.tissue_detection_models:
            if model in model_set:
                model_run_sequence.remove(model)
                model_run_sequence.insert(0, model)
                break
        logging.info(f"Prioritized tissue model in sequence: {model_run_sequence}")
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
            logging.info("Loaded tissue geom.")
        
        model = getattr(self, model_name, None)
        if model is None:
            self.list_available_models()
            logging.error(f"Model {model_name} not found in Diagnosis class.")
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

        logging.info(f"Starting inference for model: {model_name}")
        model.infer(dataloader)
