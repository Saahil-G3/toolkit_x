import warnings
import subprocess

import torch


class NoGpuDetected(UserWarning):
    pass


class GpuManager:
    def __init__(self, gpu_id=0, device_type="gpu", dataparallel=False, dataparallel_device_ids=None):
        self.gpu_id = gpu_id
        self.device_type = device_type
        self._gpu_exists = torch.cuda.is_available()
        self._gpu_count = torch.cuda.device_count()
        self._dataparallel = dataparallel
        self._dataparallel_device_ids = dataparallel_device_ids
        if self._gpu_count > 0:
            self._gpu_names = {}
            for i in range(self._gpu_count):
                self._gpu_names[i] = torch.cuda.get_device_name(i)

            if self._dataparallel_device_ids is None:
                self._dataparallel_device_ids = list(range(self._gpu_count))
                
        if self.device_type == "gpu":
            self._set_gpu(self.gpu_id)
        elif self.device_type == "cpu":
            self.device = self._get_cpu()

    def _get_gpu(self, gpu_id=0):
        gpu = torch.device(f"cuda:{gpu_id}")
        return gpu

    def _get_cpu(self):
        cpu = torch.device("cpu")
        return cpu

    def _set_gpu(self, gpu_id=0):
        if self._gpu_exists and gpu_id < self._gpu_count:
            self.device = self._get_gpu(gpu_id=gpu_id)
        else:
            self.device = self._get_cpu()
            warnings.warn("Invalid GPU ID or No GPU detected, using CPU", NoGpuDetected)

    def _get_free_gpu_memory(self, device_id=0):
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--query-gpu=memory.free,memory.total",
                    f"--format=csv,nounits,noheader",
                    f"--id={device_id}",
                ],
                encoding="utf-8",
            )
            for line in result.strip().split("\n"):
                free_memory, total_memory = line.split(",")
                free_memory = float(free_memory)
                total_memory = float(total_memory)
                percent_free = round((free_memory / total_memory) * 100, 2)

                return percent_free

        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            return None
