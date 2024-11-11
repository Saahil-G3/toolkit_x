from geometry.shapely_tools import prep_geom_for_query, get_box
from tqdm.auto import tqdm

class InitSlicer():
    def __init__(self, wsi, tissue_geom = None, sample_using_tissue_geom = False):
        
        self.wsi = wsi
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
                raise ValueError("Sampling with tissue geometry cannot be done because 'tissue_geom' is None.")
        
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

        if self.sample_using_tissue_geom:
            self.params = self.sph[self.slice_key]["params"]
            self.n_contained_coordinates = len(self.sph[self.slice_key]["filtered_coordinates"]["contained_coordinates"])
            self.n_boundary_coordinates = len(self.sph[self.slice_key]["filtered_coordinates"]["boundary_coordinates"])
            #self.total_samples = self.n_contained_coordinates+self.n_boundary_coordinates
            self.filtered_index = -1
        else:
            self.params = self.sph[self.slice_key]["params"]
            #self.total_samples = len(self.sph[self.slice_key]["coordinates"])

    def set_params(
        self,
        target_mpp,
        patch_size,
        overlap_size,
        context_size,
        slice_key=None,
        input_tuple=False,
        show_progress=False
    ):
        self.set_params_init = True
        self.default_slice_key += 1
        slice_key = slice_key or self.default_slice_key
        self.recent_slice_key = slice_key
        self.sph[slice_key] = {"params": {}}
        params = self.sph[slice_key]["params"]
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
    
        self.sph[slice_key]["coordinates"] = self.wsi._get_slice_wsi_coordinates(params)
    
        if self.tissue_geom is not None:
            self._filter_coordinates(slice_key, show_progress=show_progress)

        
    def _filter_coordinates(self, slice_key, show_progress=False):
        params = self.sph[slice_key]["params"]
        extraction_dims = params["extraction_dims"]
        factor1 = params["factor1"]
        coordinates = self.sph[slice_key]["coordinates"]
        
        box_width = extraction_dims[0] * factor1
        box_height = extraction_dims[1] * factor1
    
        contained_coordinates = []
        boundary_coordinates = []
    
        # Use tqdm only if show_progress is True
        iterator = coordinates
        if show_progress:
            iterator = tqdm(coordinates, desc='Filtering coordinates')
    
        for x, y in iterator:
            box = get_box(x, y, box_width, box_height)
    
            if self.tissue_geom_prepared.intersects(box):
                if self.tissue_geom_prepared.contains(box):
                    contained_coordinates.append((x, y))
                else:
                    boundary_coordinates.append((x, y))
    
        self.sph[slice_key]["filtered_coordinates"] = {
            "contained_coordinates": contained_coordinates,
            "boundary_coordinates": boundary_coordinates,
        }
        
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
            context_dims[0] + overlap_dims[0],
            context_dims[1] + overlap_dims[1],
        )
    
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
        
    def get_slice_region(self, params, coordinates):
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
