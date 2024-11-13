import ocv
import shapely_utils

import pandas as pd
from openslide import OpenSlide
import tiffslide as openslide

from pathlib import Path
import numpy as np
import cv2
from skimage import measure
from skimage.morphology import remove_small_holes
from torch.utils.data import Dataset as BaseDataset

def remove_objects(mask,target_mpp,area_thresh):
    
    labeled = measure.label(mask)
    cleaned_mask = np.zeros_like(mask)
    
    for region in measure.regionprops(labeled):
        
        if region.area<area_thresh:
            continue
        else:
            cleaned_mask[labeled == region.label] = 1
    
    return cleaned_mask


class WSI:
    def __init__(self, path, mpp):
        self._path=Path(path)
        self._slide=OpenSlide(self._path)
        self.stem=self._path.stem
        self.dims=self._slide.dimensions
        self.mpp=mpp
        
    def get_region(self, x, y, w, h, level=0):
        '''
        args:
            x,y,h,w,level=0
        returns:
            PIL region
        '''
        return self._slide.read_region(
            (int(x), int(y)), 
            level, 
            (int(w), int(h))
        )
        
    def get_patch(self, x, y, w, h, level=0):
        '''
        args:
            x,y,h,w,level=0
        returns:
            RGB numpy array
        '''
        return np.array(
            self._slide.read_region(
                (int(x), int(y)),
                level,
                (int(w), int(h))
                )
        )[:,:,:3]

    def get_dims_at_scale(self, scale):
        return (int((np.array(self.dims)*scale)[0]), int((np.array(self.dims)*scale)[1]))
    
    def get_thumbnail_at_dims(self, dims=(512,512)):
        return self._slide.get_thumbnail(dims)
        
    def get_thumbnail_at_mpp(self, target_mpp):
        return self._slide.get_thumbnail((self.get_dims_at_mpp(target_mpp)))

    def get_dims_at_mpp(self, target_mpp):
        scale, rescale=self.scale_mpp(target_mpp)
        scaled_dims=self.get_dims_at_scale(scale)
        return scaled_dims
    
    def scale_mpp(self, target_mpp):
        scale=self.mpp/target_mpp
        rescale=1/scale
        return scale,rescale

    def get_tissuemask_fast(self, limits=(20,180)):
        stride=256
        tissue_mpp=self.mpp * (self.dims[1] / (self.dims[1] // stride))
        thumbnail=self.get_thumbnail_at_dims(self.get_dims_at_mpp(tissue_mpp))
        thumbnail=np.array(thumbnail.convert('L'))
        #thumbnail=highPassFilter(thumbnail)
        #thumbnail=np.array(self.get_thumbnail_at_mpp(target_mpp)).astype(np.uint8)
        tissue_mask=(np.logical_and(thumbnail<limits[1], thumbnail>limits[0])).astype(np.uint8)
        tissue_mask=remove_small_holes(tissue_mask.astype(bool), area_threshold=1000).astype(np.uint8)
        tissue_mask=cv2.dilate(tissue_mask, (5,5), iterations=5)
        tissue_mask=remove_small_holes(tissue_mask.astype(bool), area_threshold=1000).astype(np.uint8)
        tissue_mask=remove_objects(tissue_mask, tissue_mpp, area_thresh=100)
        tissue_mask=cv2.dilate(tissue_mask, (5,5), iterations=2)
        tissue_mask=cv2.medianBlur(tissue_mask,11)
        return tissue_mask, tissue_mpp

    def get_tissue_polys(self, limits=(20,180)):
        tissue_mask,target_mpp=self.get_tissuemask_fast(limits=limits)
        c,h=ocv.get_contours(tissue_mask)
        scale,rescale=self.scale_mpp(target_mpp)
        wkts=ocv.process_contour_hierarchy(c,h, contour_mpp=target_mpp,rescale_factor=rescale)
        polys=[shapely_utils.loads(wkt['master_wkt']) for wkt in wkts]
        return polys

    def get_patch_polybox(self, polyBox):
        minx,miny,maxx,maxy=polyBox.bounds
        w=int(maxx - minx)
        h=int(maxy - miny)
        return self.get_patch(int(minx),int(miny),w, h)

    def get_tissuemask(self,target_mpp):
        """
        """
        img=np.array(self.get_thumbnail_at_mpp(target_mpp)).astype(np.uint8)
        BILATERAL1_ARGS = {"d": 40, "sigmaColor": 100, "sigmaSpace": 150}
        THRESH1_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU}
        THRESH2_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}
        
        img = highPassFilter(img)
        img_temp = np.where(img < 100, 255, img)
        img_hsv = cv2.cvtColor(img_temp, cv2.COLOR_RGB2HSV)
        lower_red = np.array([100, 20, 0])
        upper_red = np.array([180, 255, 255])
        mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
        
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([0, 50, 50])
        upper_blue = np.array([130, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        
        img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        blue_patch = cv2.bitwise_and(img, img, mask=mask_blue)
        img = img - blue_patch
        
        m = cv2.bitwise_and(img, img, mask=mask_red)
        im_fill = np.where(m == 0, 233, m)
        
        image = img.copy()
        shape = img.shape
        gray = cv2.cvtColor(im_fill, cv2.COLOR_BGR2GRAY)
        
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        blur_final = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    
        blur1 = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
        blur_final = cv2.bilateralFilter(blur1, **BILATERAL1_ARGS)
        blur_final = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
        
        _, thresh = cv2.threshold(blur_final, **THRESH1_ARGS)
        _, thresh = cv2.threshold(thresh, **THRESH2_ARGS)
        
        return(thresh)

class CPDataset(BaseDataset):
    def __init__(self, data_distribution:pd.DataFrame):
        super(CPDataset, self).__init__()
        self.data_distribution=data_distribution
        self.positive_cases=data_distribution[data_distribution['ann_name'].notna()]
        self.negative_cases=data_distribution[data_distribution['ann_name'].isna()]

    def sample_mixed_rows(self, n_samples=1, replace=True):
        return self.data_distribution.sample(n_samples, replace=replace).to_dict(orient='records')
        
    def sample_positive_rows(self, n_samples=1, replace=True):
        return self.positive_cases.sample(n_samples, replace=replace).to_dict(orient='records')

    def sample_negative_rows(self, n_samples=1, replace=True):
        return self.negative_cases.sample(n_samples, replace=replace).to_dict(orient='records')

    def get_wsi_object(self, sample):
        self.wsi=WSI(f"{sample['wsi_folder']}/{sample['wsi_name']}", mpp=sample['mpp'])
        
    #def get_ann_dict(self, sample):
    #    pass
    #    return load.load_pickle(f"{sample['ann_folder']}/{sample['ann_name']}")

    def process_ann_dict(self, ann_dict):
        # Raise an error to signal that this method should be implemented in a subclass or later
        raise NotImplementedError("process_ann_dict() must be implemented in a subclass.")

def extract_tissue_without_fat(img):
    
    BILATERAL1_ARGS={"d":9,"sigmaColor":10000,"sigmaSpace":150}
    BILATERAL2_ARGS={"d":90,"sigmaColor":5000,"sigmaSpace":5000}
    BILATERAL3_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":10000}
    BILATERAL4_ARGS={"d":90,"sigmaColor":10000,"sigmaSpace":100}
    
    THRESH1_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_TRUNC+cv2.THRESH_OTSU}
    THRESH2_ARGS={"thresh":0,"maxval":255,"type":cv2.THRESH_OTSU}
    
    image = img.copy()
    w = img.shape[1]
    h = img.shape[0]
    
    shape=img.shape
    
    img_hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    
    lower_red=np.array([120,0,0])
    upper_red=np.array([180,255,255])
    
    mask=cv2.inRange(img_hsv,lower_red,upper_red)
    
    img_hsv=cv2.cvtColor(img_hsv,cv2.COLOR_HSV2RGB)
    
    m=cv2.bitwise_and(img,img,mask=mask)
    
    im_fill=np.where(m==0,233,m)
    
    mask=np.zeros(shape[:2])
    
    gray=cv2.cvtColor(im_fill,cv2.COLOR_BGR2GRAY)
    blur1=cv2.bilateralFilter(np.bitwise_not(gray),**BILATERAL1_ARGS)
    blur2=cv2.bilateralFilter(np.bitwise_not(blur1),**BILATERAL2_ARGS)
    blur3=cv2.bilateralFilter(np.bitwise_not(blur2),**BILATERAL3_ARGS)
    blur4=cv2.bilateralFilter(np.bitwise_not(blur3),**BILATERAL4_ARGS)
    
    blur_final=255-blur4
    _,thresh=cv2.threshold(blur_final,**THRESH1_ARGS)
    _,thresh=cv2.threshold(thresh,**THRESH2_ARGS)
    
    return thresh


    
def extract_tissue(img):
    """
    """
    
    BILATERAL1_ARGS = {"d": 40, "sigmaColor": 100, "sigmaSpace": 150}
    THRESH1_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU}
    THRESH2_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}
    
    img = highPassFilter(img)
    img_temp = np.where(img < 100, 255, img)
    img_hsv = cv2.cvtColor(img_temp, cv2.COLOR_RGB2HSV)
    lower_red = np.array([100, 20, 0])
    upper_red = np.array([180, 255, 255])
    mask_red = cv2.inRange(img_hsv, lower_red, upper_red)
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_blue = np.array([0, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    blue_patch = cv2.bitwise_and(img, img, mask=mask_blue)
    img = img - blue_patch
    
    m = cv2.bitwise_and(img, img, mask=mask_red)
    im_fill = np.where(m == 0, 233, m)
    
    image = img.copy()
    shape = img.shape
    gray = cv2.cvtColor(im_fill, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    blur_final = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    blur1 = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
    blur_final = cv2.bilateralFilter(blur1, **BILATERAL1_ARGS)
    blur_final = cv2.bilateralFilter(blur_final, **BILATERAL1_ARGS)
    
    _, thresh = cv2.threshold(blur_final, **THRESH1_ARGS)
    _, thresh = cv2.threshold(thresh, **THRESH2_ARGS)
    
    return(thresh)

def highPassFilter(img):
    
    BILATERAL1_ARGS = {"d": 40, "sigmaColor": 100, "sigmaSpace": 150}
    THRESH1_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_TRUNC + cv2.THRESH_OTSU}
    THRESH2_ARGS = {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU}
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_f = np.float32(img_gray)
    fur_img = cv2.dft(img_f, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(fur_img)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    rows, cols = img_f.shape
    crow, ccol = rows // 2, cols // 2
    fw = 200

    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[:, :ccol - fw, :] = 1
    mask[:, ccol + fw:, :] = 1
    mask[:crow - fw, ccol - fw:ccol + fw, :] = 1
    mask[crow + fw:, ccol - fw:ccol + fw, :] = 1
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = img_back * 255 // (np.max(img_back))
    img_back = img_back.astype(int)
    img_back = img_back.astype('uint8')
    blur = cv2.GaussianBlur(img_back, (5, 5), 0)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blur = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, ker)
    er_img = cv2.erode(blur, np.ones((5, 5), np.uint8))
    er_img = cv2.erode(er_img, np.ones((5, 5), np.uint8))
    blur = cv2.bilateralFilter(er_img, **BILATERAL1_ARGS)
    blur = cv2.bilateralFilter(blur, **BILATERAL1_ARGS)
    blur = cv2.bilateralFilter(blur, **BILATERAL1_ARGS)
    _, blur = cv2.threshold(blur, **THRESH1_ARGS)
    _, blur = cv2.threshold(blur, **THRESH2_ARGS)
    img = cv2.bitwise_and(img, img, mask=blur)
    return img
