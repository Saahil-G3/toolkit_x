from torch.utils.data import DataLoader

from ._init_slicer import InitSlicer
from ._torch_datasets import _DatasetAll, _DatasetFiltered


class Slicer(InitSlicer):
    def __init__(self, wsi, device, tissue_geom=None):
        InitSlicer.__init__(self, wsi, tissue_geom=None)
        self.slice_key = None
        self.device = device

    def set_tissue_geom(self, tissue_geom):
        """
        Overriding the parent's set_tissue_geom method, if needed.
        """
        super().set_tissue_geom(tissue_geom)

    def get_torch_dataloader(
        self, dataset_type, batch_size=2, shuffle=False, num_workers=0, data_loading_mode="cpu", **kwargs
    ):

        if dataset_type == "all_coordinates":
            dataset = _DatasetAll(self, data_loading_mode)
        elif dataset_type == "filtered_coordinates":
            dataset = _DatasetFiltered(self, data_loading_mode)
        else:
            raise ValueError(f"dataset type {dataset_type} not implemented")

        if self.wsi.wsi_type == "TiffSlide" and num_workers > 0:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                worker_init_fn=dataset.worker_init,
                **kwargs,
            )
        else:
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                **kwargs,
            )
        return dataloader

    def set_slice_key(self, slice_key):
        self.slice_key = slice_key