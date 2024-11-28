from pathlib import Path

script_dir = Path(__file__).resolve().parent
toolkit_weights_dir = Path("toolkit/weights/qc/qc_models_v1")
weights_dir = None
for parent in script_dir.parents[::-1]:
    #print(parent)
    if "toolkit_x" in str(parent):
        weights_dir = parent / toolkit_weights_dir
        break
        
if weights_dir is None or not weights_dir.exists():
    raise FileNotFoundError("Could not find the 'toolkit_x' directory or the weights path does not exist.")

from .base_qc_model import BaseQCModel

class PenModelV1(BaseQCModel):
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
        
    def _set_model_specific_params(self) -> None:
        self._detects_tissue = False
        self._model_name = "pen_model_v1"
        self._class_map = {"bg": 0, "pen_mark": 1}
        self._mpp = 16
        self._med_blur_ksize = 15

    def _set_model_class(self) -> None:
        self._state_dict_path = Path(f"{weights_dir}/pen_model_v1.pt")
        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet34"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 2

class FocusModelV1(BaseQCModel):
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
        
    def _set_model_specific_params(self) -> None:
        self._detects_tissue = False
        self._model_name = "focus_model_v1"
        self._class_map = {
            "bg": 0,
            "level_1": 1,
            "level_2": 2,
            "level_3": 3,
            "level_4": 4,
        }
        self._mpp = 2
        self._med_blur_ksize = 15

    def _set_model_class(self) -> None:
        self._state_dict_path = Path(f"{weights_dir}/focus_model_v1.pt")
        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet18"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 5

class FoldsModelV1(BaseQCModel):
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
        
    def _set_model_specific_params(self) -> None:
        self._detects_tissue = False
        self._model_name = "folds_model_v1"
        self._class_map = {"bg": 0, "fold": 1}
        self._mpp = 2
        self._med_blur_ksize = 15

    def _set_model_class(self) -> None:
        self._state_dict_path = Path(f"{weights_dir}/folds_model_v1.pt")
        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet18"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 2