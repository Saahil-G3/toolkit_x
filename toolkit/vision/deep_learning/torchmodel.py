import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from toolkit.system.gpu.torch import GpuManager

from toolkit.system.logging_tools import Logger
logger = Logger(name="torchmodel").get_logger()

class _BaseModel(GpuManager):
    def __init__(
        self,
        gpu_id=0,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
    ):
        GpuManager.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        self.model = None

    def load_model(self, replace_model=False):
        if self.model is None or replace_model:
            if self._model_class == "smp":
                self._load_smp_model()
                if self.device_type == "gpu":
                    logger.info(f"Loaded model {self.model_name} on {self.device_type.upper()}:{self.gpu_id}")
                else:
                    logger.info(f"Loaded model {self.model_name} on {self.device_type.upper()}")
            else:
                raise ValueError(f"model class{self.model_class} not implemented ")
        else:
            logger.warning("Model already loaded, set replace_model=True for replacing the current model")

    def _load_smp_model(self):
        if self._architecture == "UnetPlusPlus":
            self.model = smp.UnetPlusPlus(
                encoder_name=self._encoder_name,
                encoder_weights=self._encoder_weights,
                in_channels=self._in_channels,
                classes=self._classes,
            )

        else:
            raise ValueError(f"Architecture {self._architecture} not implemented ")

        self.model.load_state_dict(
            torch.load(
                self._state_dict_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        self.model.to(self.device)
        if self._dataparallel:
            self.model = nn.DataParallel(
                self.model, device_ids=self._dataparallel_device_ids
            )
