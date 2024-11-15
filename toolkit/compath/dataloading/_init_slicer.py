from pathlib import Path
from tqdm.auto import tqdm

from toolkit.compath.slide.wsi import WSIManager
from toolkit.geometry.shapely_tools import prep_geom_for_query, get_box
from toolkit.system.gpu.torch import GpuManager


class InitSlicer(GpuManager):
    def __init__(
        self,
        gpu_id=0,
        tissue_geom=None,
        device_type="gpu",
        dataparallel=False,
        dataparallel_device_ids=None,
        sample_using_tissue_geom=False,
    ):
        GpuManager.__init__(
            self,
            gpu_id=gpu_id,
            device_type=device_type,
            dataparallel=dataparallel,
            dataparallel_device_ids=dataparallel_device_ids,
        )
        
        self.gpu_id = gpu_id
        if self.device_type == "gpu":
            self._set_gpu(self.gpu_id)
        elif self.device_type == "cpu":
            self.device = self._get_cpu()
        
        self.sph = {}  # Slice Parameters History
        self.default_slice_key = -1
        self.set_params_init = False
        self.recent_slice_key = None
        self.tissue_geom = tissue_geom
        self.sample_using_tissue_geom = sample_using_tissue_geom

        if self.tissue_geom is not None:
            self.sample_using_tissue_geom = True
        else:
            if self.sample_using_tissue_geom:
                raise ValueError(
                    "Sampling with tissue geometry cannot be done because 'tissue_geom' is None."
                )

    def set_wsi(self, wsi_path, wsi_type):
        self.wsi = WSIManager(wsi_path).get_wsi_object(wsi_type)
        
    def set_tissue_geom(self, tissue_geom):
        self.tissue_geom = tissue_geom
        self.tissue_geom_prepared = prep_geom_for_query(tissue_geom)
        self.sample_using_tissue_geom = True

    def set_slicer(self, slice_key=None):
        if not self.set_params_init:
            raise ValueError("No slice parameters have been initialized")

        if slice_key is None:
            self.slice_key = self.recent_slice_key
        else:
            self.slice_key = slice_key

        self.params = self.sph[self.slice_key]["params"]

    def set_params(
        self,
        target_mpp:float,
        patch_size,
        overlap_size,
        context_size,
        slice_key=None,
        input_tuple: bool =False,
        show_progress: bool =False,
    ):
        self.set_params_init = True
        self.default_slice_key += 1
        slice_key = slice_key or self.default_slice_key
        self.recent_slice_key = slice_key
        self.sph[slice_key] = {"params": {}}
        params = self.sph[slice_key]["params"]
        self.sph[slice_key]["wsi_name"] = Path(self.wsi._wsi_path.name)
        params["target_mpp"] = target_mpp

        if input_tuple:
            params["patch_dims"] = patch_dims
            params["overlap_dims"] = overlap_dims
            params["context_dims"] = context_dims
        else:
            params["patch_dims"] = (patch_size, patch_size)
            params["overlap_dims"] = (overlap_size, overlap_size)
            params["context_dims"] = (context_size, context_size)

        self._set_params(slice_key=slice_key)

        self.sph[slice_key]["all_coordinates"] = self.wsi._get_slice_wsi_coordinates(params)

        if self.tissue_geom is not None:
            self._filter_coordinates(slice_key, show_progress=show_progress)

    def _filter_coordinates(self, slice_key, show_progress=False):
        params = self.sph[slice_key]["params"]
        extraction_dims = params["extraction_dims"]
        factor1 = params["factor1"]
        coordinates = self.sph[slice_key]["all_coordinates"]

        box_width = extraction_dims[0] * factor1
        box_height = extraction_dims[1] * factor1

        tissue_contact_coordinates = []  # (coordinate, is_boundary)

        iterator = coordinates
        if show_progress:
            iterator = tqdm(coordinates, desc="Filtering coordinates")

        for (x, y), _ in iterator:
            box = get_box(x, y, box_width, box_height)

            if self.tissue_geom_prepared.intersects(box):
                if self.tissue_geom_prepared.contains(box):
                    tissue_contact_coordinates.append(((x, y), False))
                else:
                    tissue_contact_coordinates.append(((x, y), True))

        self.sph[slice_key]["tissue_contact_coordinates"] = tissue_contact_coordinates
        self.sample_using_tissue_geom = True

    def _set_params(self, slice_key):
        """
        factor1 : factor to downsample from original_mpp to target_mpp
        factor2 : factor to downsample from original_mpp to downsample_mpp
        factor3 : factor to downsample from downsample_mpp to target_mpp
        """

        params = self.sph[slice_key]["params"]

        context_dims = params["context_dims"]
        overlap_dims = params["overlap_dims"]
        patch_dims = params["patch_dims"]

        params["shift_dims"] = (
            context_dims[0] + overlap_dims[0]//2,
            context_dims[1] + overlap_dims[1]//2,
        )
        
        params["shift_dims"] = (overlap_dims[0], overlap_dims[1])

        params["extraction_dims"] = (
            patch_dims[0] + 2 * context_dims[0],
            patch_dims[1] + 2 * context_dims[1],
        )

        params["stride_dims"] = (
            patch_dims[0] - (overlap_dims[0] + 2 * context_dims[0]),
            patch_dims[1] - (overlap_dims[1] + 2 * context_dims[1]),
        )

        params["factor1"] = self.wsi.factor_mpp(params["target_mpp"])
        params["level"] = self.wsi.get_level_for_downsample(params["factor1"])
        params["level_dims"] = self.wsi.get_level_dimensions()[params["level"]]
        params["mpp_at_level"] = (
            self.wsi.get_level_downsamples()[params["level"]] * self.wsi.mpp
        )
        params["factor2"] = self.wsi.factor_mpp(params["mpp_at_level"])
        params["factor3"] = self.wsi.factor_mpp(
            target_mpp=params["target_mpp"], source_mpp=params["mpp_at_level"]
        )

        extraction_dims = params["extraction_dims"]
        factor3 = params["factor3"]
        params["extraction_dims_at_level"] = (
            self.round_to_nearest_even(extraction_dims[0] * factor3),
            self.round_to_nearest_even(extraction_dims[1] * factor3),
        )

        stride_dims = params["stride_dims"]
        params["stride_dims_at_level"] = (
            self.round_to_nearest_even(stride_dims[0] * factor3),
            self.round_to_nearest_even(stride_dims[1] * factor3),
        )

    def get_slice_region(self, coordinates, params):
        region = self.wsi.get_region(
            coordinates[0],
            coordinates[1],
            params["extraction_dims_at_level"][0],
            params["extraction_dims_at_level"][1],
            params["level"],
        )
        return region

    @staticmethod
    def round_to_nearest_even(x):
        return round(x / 2) * 2
