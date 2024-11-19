from pathlib import Path

script_dir = Path(__file__).parent
weights_dir = script_dir / "weights"

from typing import Optional, List

from .base_qc_model import BaseQCModel

"""
Arguments Example:
  
node_detection_v1 = {
    "gpu_id": 0,
    "patch_size": 256,
    "device_type": "gpu",
    "dataparallel": None
}

tissue_model_v1 = {
    "gpu_id": 0,
    "patch_size": 256,
    "device_type": "gpu",
    "dataparallel": None
}

folds_model_v1 = {
    "gpu_id": 0,
    "patch_size": 256,
    "device_type": "gpu",
    "dataparallel": None
}

focus_model_v1 = {
    "gpu_id": 0,
    "patch_size": 256,
    "device_type": "gpu",
    "dataparallel": None
}

pen_model_v1 = {
    "gpu_id": 0,
    "patch_size": 256,
    "device_type": "gpu",
    "dataparallel": None
}
"""


class NodeDetectionV1(BaseQCModel):
    def __init__(
        self,
        gpu_id: int = 0,
        patch_size: int = 256,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: Optional[List[int]] = None,
    ):
        BaseQCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.model_name = "node_detection_v1"
        self.state_dict_path = Path(f"{weights_dir}/node_detection_v1.pth")
        self.class_map = {"bg": 0, "node": 1}
        self.mpp = 3.2

        self.patch_size = patch_size
        self.overlap_size = int(self.patch_size * 0.0625)
        self.context_size = self.overlap_size
        
        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet34"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 2


class PenModelV1(BaseQCModel):
    def __init__(
        self,
        gpu_id: int = 0,
        patch_size: int = 512,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
    ):
        BaseQCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.model_name = "pen_model_v1"
        self.state_dict_path = Path(f"{weights_dir}/pen_model_v1.pt")
        self.class_map = {"bg": 0, "pen_mark": 1}
        self.mpp = 16

        self.patch_size = patch_size
        self.overlap_size = int(self.patch_size * 0.0625)
        self.context_size = self.overlap_size

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
        patch_size: int = 1024,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
    ):
        BaseQCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.model_name = "focus_model_v1"
        self.state_dict_path = Path(f"{weights_dir}/focus_model_v1.pt")
        self.class_map = {
            "bg": 0,
            "level_1": 1,
            "level_2": 2,
            "level_3": 3,
            "level_4": 4,
        }
        self.mpp = 2

        self.patch_size = patch_size
        self.overlap_size = int(self.patch_size * 0.0625)
        self.context_size = self.overlap_size

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
        patch_size: int = 1024,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
    ):
        BaseQCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.model_name = "folds_model_v1"
        self.state_dict_path = Path(f"{weights_dir}/folds_model_v1.pt")
        self.class_map = {"bg": 0, "fold": 1}
        self.mpp = 2

        self.patch_size = patch_size
        self.overlap_size = int(self.patch_size * 0.0625)
        self.context_size = self.overlap_size

        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet18"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 2


class TissueModelV1(BaseQCModel):
    def __init__(
        self,
        gpu_id: int = 0,
        patch_size: int = 1024,
        device_type: str = "gpu",
        dataparallel: bool = False,
        dataparallel_device_ids: list[int] = None,
    ):
        BaseQCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )

        self.model_name = "tissue_model_v1"
        self.state_dict_path = Path(f"{weights_dir}/tissue_model_v1.pt")
        self.class_map = {"bg": 0, "adipose": 1, "non_adipose": 2}
        self.mpp = 4

        self.patch_size = patch_size
        self.overlap_size = int(self.patch_size * 0.0625)
        self.context_size = self.overlap_size

        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet18"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 3
