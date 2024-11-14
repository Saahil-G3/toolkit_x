from pathlib import Path
from .base_model_class import QCModel

script_dir = Path(__file__).parent
weights_dir = script_dir/"weights"

class TissueModel(QCModel):
    def __init__(
        self,
        patch_size=1024,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        QCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.model_name = "tissue"
        self.state_dict_path = Path(f"{weights_dir}/tissue.pt")
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

class FoldsModel(QCModel):
    def __init__(
        self,
        patch_size=1024,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        QCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.model_name = "folds"
        self.state_dict_path = Path(f"{weights_dir}/folds.pt")
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

class FocusModel(QCModel):
    def __init__(
        self,
        patch_size=1024,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        QCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.model_name = "focus"
        self.state_dict_path = Path(f"{weights_dir}/focus.pt")
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

class PenModel(QCModel):
    def __init__(
        self,
        patch_size=256,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        QCModel.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.model_name = "pen"
        self.state_dict_path = Path(f"{weights_dir}/pen.pt")
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
