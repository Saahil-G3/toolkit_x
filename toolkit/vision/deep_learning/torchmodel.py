import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from toolkit.gpu_tools.torch import GpuManager


class Model(GpuManager):
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
            
    def load_model(self):
        if self._model_class == "smp":
            self._load_smp_model()
        else:
            raise ValueError(f"model class{self.model_class} not implemented ")

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
                self.state_dict_path,
                map_location=self.device,
                weights_only=True,
            )
        )

        if self.dataparallel:     
            self.model = nn.DataParallel(
                self.model, device_ids=self.dataparallel_device_ids
            )
            self.model.to(self.device)




