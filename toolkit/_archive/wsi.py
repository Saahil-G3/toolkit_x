import cv2
import warnings
import numpy as np
from pathlib import Path
from openslide import OpenSlide
from tiffslide import TiffSlide


from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset

from misc import round_to_nearest_even

class InitWSI():
    def __init__(self, wsi_path, tissue_geom=None, mpp=None):

        self.wsi_path = Path(wsi_path)
        self.wsi = OpenSlide(self.wsi_path)
        self.tissue_geom = tissue_geom
        self.mpp = mpp
        
        if self.wsi.level_count==1:
            self.wsi = TiffSlide(wsi_path)
            self.wsi_type='TS'
            if self.wsi.level_count==1:
                print(f"unable to open with tiffslide as well, please check.")
        else:
            self.wsi_type='OS'

        self.dims=self.wsi.dimensions
        self.get_mpp()
        self.level_count = self.wsi.level_count
        
    def get_dims_at_mpp(self, target_mpp):
        scale, rescale=self.scale_mpp(target_mpp)
        scaled_dims=self.get_dims_at_scale(scale)
        return scaled_dims

    def get_dims_at_scale(self, scale):
        return (
            int((np.array(self.dims) * scale)[0]),
            int((np.array(self.dims) * scale)[1]),
        )

    def get_thumbnail_at_mpp(self, target_mpp=50):
        return self.wsi.get_thumbnail(self.get_dims_at_mpp(target_mpp))

    def get_thumbnail_at_dims(self, dims):
        return self.wsi.get_thumbnail(dims)

    def factor_mpp(self, target_mpp, source_mpp=None):
        
        if source_mpp is None:
            factor=target_mpp/self.mpp
        else:
            factor=target_mpp/source_mpp
        return factor

    def _scale_mpp(self, target_mpp):
        
        scale=self.mpp/target_mpp
        rescale=1/scale
        return scale,rescale

    def scale_mpp(self, target_mpp):

        rescale = self.factor_mpp(target_mpp)
        scale = 1/rescale
        return scale,rescale
        
    def get_mpp(self):
        if self.mpp is None:
            if self.wsi_type=='TS':
                mpp_x = self.wsi.properties.get('tiffslide.mpp-x')
                mpp_y = self.wsi.properties.get('tiffslide.mpp-y')
                self.mpp=mpp_x
                if mpp_x!=mpp_y:
                    warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)
        
            elif self.wsi_type=='OS':
                mpp_x = self.wsi.properties.get('openslide.mpp-x')
                mpp_y = self.wsi.properties.get('openslide.mpp-y')
                self.mpp=mpp_x
                if mpp_x!=mpp_y:
                    warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)
                    
            else:
                self.mpp=None
        
            if self.mpp==None:
                raise ValueError("unable to calculate mpp, provide manually,")

    def get_region(self, x, y ,w, h, level):
        return self.wsi.read_region(
            (int(x), int(y)),
            level,
            (int(w), int(h))
        )

    def set_slice_wsi_coordinates(
        self, target_mpp, patch_dims, overlap_dims, context_dims
    ):
        """
        factor1 : factor to downsample from original_mpp to target_mpp
        factor2 : factor to downsample from original_mpp to downsample_mpp
        factor3 : factor to downsample from downsample_mpp to target_mpp
        """
    
        self._target_mpp = target_mpp
        self._patch_dims = patch_dims
        self._overlap_dims = overlap_dims
        self._context_dims = context_dims
        self._shift_dims = (
            self._context_dims[0] + self._overlap_dims[0],
            self._context_dims[1] + self._overlap_dims[1],
        )
    
        self._region_dims = (
            self._patch_dims[0] + 2 * self._context_dims[0],
            self._patch_dims[1] + 2 * self._context_dims[1],
        )
        self._step_dims = (
            self._patch_dims[0]
            - (self._overlap_dims[0] + 2 * self._context_dims[0]),
            self._patch_dims[1]
            - (self._overlap_dims[1] + 2 * self._context_dims[1]),
        )
    
        self._factor1 = self.factor_mpp(self._target_mpp)
        self._level = self.wsi.get_best_level_for_downsample(self._factor1)
        self._level_dims = self.wsi.level_dimensions[self._level]
        x_lim, y_lim = self._level_dims
        downsample_mpp = self.wsi.level_downsamples[self._level] * self.mpp
        self._factor2 = self.factor_mpp(downsample_mpp)
        self._factor3 = self.factor_mpp(
            target_mpp=target_mpp, source_mpp=downsample_mpp
        )
    
        self._downsample_region_dims = (
            round_to_nearest_even(self._region_dims[0] * self._factor3),
            round_to_nearest_even(self._region_dims[1] * self._factor3),
        )
    
        self._downsample_step_dims = (
            round_to_nearest_even(self._step_dims[0] * self._factor3),
            round_to_nearest_even(self._step_dims[1] * self._factor3),
        )
    
        self._coordinates = []
    
        for x in range(
            -self._context_dims[0],
            x_lim + self._context_dims[0],
            self._downsample_step_dims[0],
        ):
            if x + self._downsample_region_dims[0] > x_lim + self._context_dims[0]:
                x = (x_lim + self._context_dims[0]) - self._downsample_region_dims[0]
    
            for y in range(
                -self._context_dims[1],
                y_lim + self._context_dims[1],
                self._downsample_step_dims[1],
            ):
                if y + self._downsample_region_dims[1] > y_lim + self._context_dims[1]:
                    y = (y_lim + self._context_dims[1]) - self._downsample_region_dims[1]
    
                x_scaled, y_scaled = int(np.floor(x * self._factor2)), int(
                    np.floor(y * self._factor2)
                )
    
                self._coordinates.append((x_scaled, y_scaled))

                
    def set_slice_tissue_coordinates(
        self, tissue_geom, target_mpp, patch_dims, overlap_dims, context_dims
    ):
        self._target_mpp = target_mpp
        self._patch_dims = patch_dims
        self._overlap_dims = overlap_dims
        self._context_dims = context_dims
        self._shift_dims = (
            self._context_dims[0] + self._overlap_dims[0],
            self._context_dims[1] + self._overlap_dims[1],
        )
    
        self._region_dims = (
            self._patch_dims[0] + 2 * self._context_dims[0],
            self._patch_dims[1] + 2 * self._context_dims[1],
        )
        self._step_dims = (
            self._patch_dims[0] - (self._overlap_dims[0] + 2 * self._context_dims[0]),
            self._patch_dims[1] - (self._overlap_dims[1] + 2 * self._context_dims[1]),
        )
        self._factor1 = self.factor_mpp(self._target_mpp)
        self._level = self.wsi.get_best_level_for_downsample(self._factor1)
        self._level_dims = self.wsi.level_dimensions[self._level]
        x_lim, y_lim = self._level_dims
        downsample_mpp = self.wsi.level_downsamples[self._level] * self.mpp
        self._factor2 = self.factor_mpp(downsample_mpp)
        self._factor3 = self.factor_mpp(target_mpp=target_mpp, source_mpp=downsample_mpp)
    
        self._downsample_region_dims = (
            round_to_nearest_even(self._region_dims[0] * self._factor3),
            round_to_nearest_even(self._region_dims[1] * self._factor3),
        )
    
        self._downsample_step_dims = (
            round_to_nearest_even(self._step_dims[0] * self._factor3),
            round_to_nearest_even(self._step_dims[1] * self._factor3),
        )
        
        self._all_coordinates = []
        self._coordinates = []
        self._boxes = []
        
        for x in range(
            -self._context_dims[0],
            x_lim + self._context_dims[0],
            self._downsample_step_dims[0],
        ):
            if x + self._downsample_region_dims[0] > x_lim + self._context_dims[0]:
                x = (x_lim + self._context_dims[0]) - self._downsample_region_dims[0]
    
            for y in range(
                -self._context_dims[1],
                y_lim + self._context_dims[1],
                self._downsample_step_dims[1],
            ):
                if y + self._downsample_region_dims[1] > y_lim + self._context_dims[1]:
                    y = (y_lim + self._context_dims[1]) - self._downsample_region_dims[1]
    
                x_scaled, y_scaled = int(np.floor(x * self._factor2)), int(
                    np.floor(y * self._factor2)
                )
                
                box = get_box(x_scaled, y_scaled ,self._patch_dims[1]*self._factor1, self._patch_dims[0]*self._factor1)
    
                self._all_coordinates.append((x_scaled, y_scaled))
                
                if self.tissue_geom.intersects(box):
                    self._coordinates.append((x_scaled, y_scaled))
                    self._boxes.append(box)
                    
    '''
    def set_slice_coordinates(self, target_mpp, patch_size, overlap):

        self._target_mpp = target_mpp 
        self._patch_size = patch_size
        self._overlap = overlap
    
        self._factor1 = self.factor_mpp(self._target_mpp)
        self._level = self.wsi.get_best_level_for_downsample(self._factor1)
        self._level_dims = self.wsi.level_dimensions[self._level]
        downsample_mpp = self.wsi.level_downsamples[self._level]*self.mpp
        self._factor2 = self.factor_mpp(downsample_mpp)
        self._factor3 = self.factor_mpp(target_mpp=target_mpp, source_mpp=downsample_mpp)
        
        self._downsample_patch_size =  round_to_nearest_even(self._patch_size*self._factor3)
        self._downsample_overlap =  round_to_nearest_even(overlap*self._factor3)
        self._step_size = self._downsample_patch_size - self._downsample_overlap
    
        x_lim, y_lim = self._level_dims
        #n_regions = len(range(0, x_lim, step_size))*len(range(0, y_lim, step_size))
        #pbar = tqdm(total=n_regions, desc='extracting regions')
        #regions = []
        self._coordinates = []
        for x in range(0, x_lim, self._step_size):
            if x+self._downsample_patch_size>x_lim:
                x = x_lim-self._downsample_patch_size
            for y in range(0, y_lim, self._step_size):
                if y+self._downsample_patch_size>y_lim:
                    y = y_lim-self._downsample_patch_size
                x_scaled, y_scaled = int(np.floor(x*self._factor2)), int(np.floor(y*self._factor2))
                self._coordinates.append((x_scaled, y_scaled))
                
        return
    '''
        
    def _get_sliced_region(self, idx):
        x, y = self._coordinates[idx]
        region = self.get_region(
            x, y, self._downsample_region_dims[0], self._downsample_region_dims[1], self._level
        )
        region = np.array(region)
        region = cv2.resize(region, (self._region_dims[1], self._region_dims[0]))
    
        return region



class SliceInferenceWSI(BaseDataset):
    def __init__(self, wsi: InitWSI):
        super(SliceInferenceWSI, self).__init__()
        self.wsi = wsi

    def __len__(self):
        return len(self.wsi._coordinates)

    def __getitem__(self, idx):
        region = np.array(self.wsi._get_sliced_region(idx))[:, :, :3]
        region = region.transpose(2, 0, 1)
        return region

    def get_dataloader(
        self,
        dataset,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        **kwargs,
    ):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
            **kwargs,
        )
        return dataloader








    