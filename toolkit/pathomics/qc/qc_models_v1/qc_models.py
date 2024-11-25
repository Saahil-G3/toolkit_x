from pathlib import Path

script_dir = Path(__file__).parent
weights_dir = script_dir / "weights"

from .base_qc_model import BaseQCModel

class NodeDetectionV1(BaseQCModel):
    def __init__(
        self,
        gpu_id: int = 0,
        patch_size=1024,
        overlap_size = None,
        context_size = None,
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

        self.detects_tissue = True
        self.model_name = "node_detection_v1"
        self._state_dict_path = Path(f"{weights_dir}/node_detection_v1.pth")
        self._class_map = {"bg": 0, "node": 1}
        self._mpp = 3.2

        self._patch_size = patch_size
        
        if overlap_size:
            self._overlap_size = overlap_size
        else:
            self._overlap_size = int(self._patch_size * 0.0625)
        
        if context_size:
            self._context_size = context_size
        else:
            self._context_size = 2 * self._overlap_size

        self._model_class = "smp"
        self._architecture = "UnetPlusPlus"
        self._encoder_name = "resnet34"
        self._encoder_weights = "imagenet"
        self._in_channels = 3
        self._classes = 2