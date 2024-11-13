'''
for documentaion refer to - 
https://docs.pathomation.com/sdk/pma.python.documentation/pma_python.html
'''

import sys

from pma_python import core
from PIL import Image
from tqdm.auto import tqdm
import numpy as np

from json.decoder import JSONDecodeError

import cv2
import image_utils as iu
#from cpath import extract_tissue_without_fat, extract_tissue
import geometry.ocv
from geometry.shapely_utils import loads, remove_duplicates_valid, MultiPolygon, Polygon

Image.MAX_IMAGE_PIXELS = None

default_layerID = 11
tissue_no_fat_layerID = 22
tissue_layerID = 33
 

def get_tray_format(slideRefs):
    
    tray_format = []

    for slideRef in slideRefs:
        temp_dict = {}
        temp_dict['Slide Info::Server;Slide Info::File name'] = 'PMA.core;'+slideRef
        tray_format.append(temp_dict)
    
    return tray_format


class PMA_Slide:
    def __init__(self,slideRef, sessionID=None):
        self.sessionID = sessionID
        self.slideRef = slideRef
        self.slideName = slideRef.split('/')[-1].split('.')[0]
        self.total_layers = core.get_number_of_layers(slideRef, sessionID=self.sessionID)
        self.file_extension = core.get_slide_file_extension(slideRef)
        self.total_zoom_levels = core.get_max_zoomlevel(slideRef, sessionID=self.sessionID)
        self.mpp = core.get_pixels_per_micrometer(slideRef, zoomlevel=None, sessionID=self.sessionID)
        self.pixel_dimensions = core.get_pixel_dimensions(self.slideRef, zoomlevel=None, sessionID=self.sessionID)
        self.mpp_zoom_mapping = self.get_mpp_zoom_mapping()
        self.fingerprint = core.get_fingerprint(self.slideRef, sessionID=self.sessionID, verify=True)
        
        if len(self.mpp) == 2:
            if round(self.mpp[0],3) == round(self.mpp[1],3):
                self.mpp = self.mpp[0]
            else:
                print('mppx is not equal to mppy')
            
    
    def get_mag_at_zoomlevel(self, zoom_level):
        mag_at_zoomlevel = core.get_magnification(self.slideRef, zoomlevel=zoom_level, exact=False, sessionID=self.sessionID)
        print(f"The magnification at zoomlevel {zoom_level} is {mag_at_zoomlevel}")
        return(mag_at_zoomlevel)
    
    def get_mpp_zoom_mapping(self):
        total_zoom_levels = core.get_max_zoomlevel(self.slideRef, sessionID=self.sessionID)
        zl_list = []
        for zl in range(total_zoom_levels+1):
            mpp_at_zl = core.get_pixels_per_micrometer(self.slideRef, zoomlevel=zl, sessionID=self.sessionID)
            mag_at_zl = core.get_magnification(self.slideRef, zoomlevel=zl, exact=True, sessionID=self.sessionID)
            if len(mpp_at_zl) ==2:
                zl_list.append({'Magnification':f"{mag_at_zl}x", 'mpp-x': mpp_at_zl[1], 'mpp-y':mpp_at_zl[0]})
            else:
                zl_list.append({'Magnification':f"{mag_at_zl}x", 'mpp': mpp_at_zl})
        return(zl_list)
        
    
    def get_mpp_at_zoomlevel(self, zoom_level):
        mpp_at_zoomlevel = core.get_pixels_per_micrometer(self.slideRef, zoomlevel=zoom_level, sessionID=self.sessionID)
        if mpp_at_zoomlevel[0] == mpp_at_zoomlevel[1]:
            mpp_at_zoomlevel = mpp_at_zoomlevel[0]
            print(f"The mpp at zoom level {zoom_level} is {mpp_at_zoomlevel}")
            return(mpp_at_zoomlevel)
        else:
            print(f"The mppx is not equal to mppy at zoom level {zoom_level}")
            print(f"The mpp at zoom level {zoom_level} is {mpp_at_zoomlevel}")
            return(mpp_at_zoomlevel)
        
    def get_pixeldim_at_zoomlevel(self, zoom_level):
        w,h = core.get_pixel_dimensions(self.slideRef, zoomlevel=zoom_level, sessionID=self.sessionID)
        return(w,h)

    def get_wsi(self, target_mpp = 2):
        """
        Returns: Thumbnail at selected resolution, default 2mpp.
        """

        scale, rescale = iu.scale_mpp(self.mpp, target_mpp)
        
        return(
            core.get_region(self.slideRef,
                            x=0, y=0,
                            width=self.pixel_dimensions[0], height=self.pixel_dimensions[1],
                            scale=scale,
                            sessionID=self.sessionID,
                            format='jpg',
                            quality=100)
              )
    
    def get_anns(self):
        annotations  = core.get_annotations(self.slideRef, sessionID=self.sessionID)
        annotation_metadata = []
    
        for annotation in annotations:
            geometry = annotation['Geometry']
            
            try:    
                #contour =  get_pma_contour(geometry)
                poly = loads(geometry)
                annotation['poly'] = poly
                minx, miny, maxx, maxy = poly.bounds
                
                h = int(maxy-miny)
                w = int(maxx-minx)
                
                annotation['maxx'] = maxx
                annotation['maxy'] = maxy
                annotation['minx'] = minx
                annotation['miny'] = miny
                annotation['h'] = h
                annotation['w'] = w
                
            except Exception as e:
                print(e)
                
            annotation_metadata.append(annotation)
            
        #print(f"{len(annotation_metadata)} annotation(s) found.")
        return(annotation_metadata)

    def check_tissue_mask_status(self):
        anns = self.get_anns()
        mask_status_dict = {}
        mask_status_dict['mask_exists']= False
        mask_status_dict['fat'] = False #If Whole tissue mask exist or not
        mask_status_dict['no_fat'] = False #If Tissue without fat exists or not
        
        for ann in anns:
            if ann['LayerID'] == tissue_no_fat_layerID:
                if ann['Classification'] == 'Tissue_w/o_Fat':
                    mask_status_dict['mask_exists'] = True
                    mask_status_dict['no_fat'] = True
                    continue
    
            if ann['LayerID'] == tissue_layerID:
                if ann['Classification'] == 'Tissue':
                    mask_status_dict['mask_exists'] = True
                    mask_status_dict['fat'] = True

        return mask_status_dict
        
    def _get_tissue_mask_without_fat(self, filter_area=1, upload=True, return_output=False, target_mpp=2):
        """
        
        filter_area = 1000000 (For Node)
        args:
            target_mpp: mpp at which the wsi is to be extracted from pma.
        returns:
            master_wkt_list(at original mpp), tissue_mask(at target mpp)
        """
        scale, rescale = iu.scale_mpp(self.mpp, target_mpp)
        wsi_thumbnail = self.get_wsi(target_mpp)
        tissue_mask = extract_tissue_without_fat(np.array(wsi_thumbnail))
        tissue_mask = cv2.resize(tissue_mask, self.pixel_dimensions)
        tissue_contours, hierarchy = ocv.get_contours(tissue_mask)
        master_wkt_list = ocv.process_contour_hierarchy(tissue_contours, hierarchy, self.mpp, rescale_factor = rescale)
        
        if upload:
            for wkt_dict in master_wkt_list:
                if wkt_dict['area']>filter_area:
                    try:
                        self.add_annotation(wkt_dict['master_wkt'],
                                            classification='Tissue_w/o_Fat',
                                            color = '#000000',
                                            notes = wkt_dict['area'],
                                            layerID = tissue_no_fat_layerID)
                    except JSONDecodeError:
                        poly = loads(wkt_dict['master_wkt'])
                        poly = remove_duplicates_valid(poly)
                        self.add_annotation(poly.wkt,classification='Tissue_w/o_Fat',color = '#000000',layerID = tissue_no_fat_layerID)
                    except Exception as e:
                        print(e)
                        continue 
        
        #if upload:
        #    pma_upload_args = {}
        #    pma_upload_args['classification'] = 'Tissue_w/o_Fat'
        #    pma_upload_args['notes'] = 'Tissue_w/o_Fat'
        #    pma_upload_args['color'] = '#000000'
        #    pma_upload_args['layer_id'] = tissue_no_fat_layerID 
        #    for wkt_dict in master_wkt_list:
        #        if wkt_dict['master_contour_area']>filter_area:
        #            try:
        #                pma_upload_args['anns'] = wkt_dict['master_wkt']
        #                self.add_annotation(args = pma_upload_args)
        #            except JSONDecodeError:    
        #                poly = Shapely.loads(wkt_dict['master_wkt'])
        #                poly = Shapely.remove_duplicates(poly)
        #                wkt_dict['master_wkt'] = poly.wkt
        #                pma_upload_args['anns'] = poly.wkt
        #                self.add_annotation(args = pma_upload_args)    
        #           except Exception as e:
        #                print(e)
        #                continue
        if return_output:
            return master_wkt_list, tissue_mask

    def _get_tissue_mask(self, filter_area=1, upload=True, return_output=False, target_mpp=2):
        '''
        args:
            target_mpp: mpp at which the wsi is to be extracted from pma.
        returns:
            master_wkt_list(at original mpp), tissue_mask(at target mpp)
        '''
        scale, rescale = iu.scale_mpp(self.mpp, target_mpp)
        wsi_thumbnail = self.get_wsi(target_mpp)
        tissue_mask = extract_tissue(np.array(wsi_thumbnail))
        #tissue_mask = cv2.resize(tissue_mask, self.pixel_dimensions)
        tissue_contours, hierarchy = ocv.get_contours(tissue_mask)
        master_wkt_list = ocv.process_contour_hierarchy(tissue_contours, hierarchy, self.mpp, rescale_factor=rescale)
        
        if upload:
            for wkt_dict in master_wkt_list:
                if wkt_dict['area']>filter_area:
                    try:
                        self.add_annotation(wkt_dict['master_wkt'],
                                            classification='Tissue',
                                            color ='#000000',
                                            notes = wkt_dict['area'],
                                            layerID=tissue_layerID)
                    except JSONDecodeError:
                        poly = loads(wkt_dict['master_wkt'])
                        poly = remove_duplicates_valid(poly)
                        self.add_annotation(poly.wkt,classification='Tissue',color = '#000000',layerID=tissue_layerID)
                    except Exception as e:
                        print(e)
                        continue
        
        #if upload:
        #    pma_upload_args = {}
        #    pma_upload_args['classification'] = 'Tissue'
        #    pma_upload_args['notes'] = 'Tissue'
        #    pma_upload_args['color'] = '#000000'
        #    pma_upload_args['layer_id'] = tissue_layerID
        #    for wkt_dict in master_wkt_list:
        #        if wkt_dict['master_contour_area']>filter_area:
        #            try:
        #                pma_upload_args['anns'] = wkt_dict['master_wkt']
        #                self.add_annotation(args = pma_upload_args)
        #            except JSONDecodeError:   
        #                poly = Shapely.loads(wkt_dict['master_wkt'])
        #                poly = Shapely.remove_duplicates(poly)
        #                wkt_dict['master_wkt'] = poly.wkt
        #                pma_upload_args['anns'] = poly.wkt
        #                self.add_annotation(args = pma_upload_args)   
        #            except Exception as e:
        #                print(e)
        #                continue
    
        if return_output:
            return master_wkt_list, tissue_mask

    def get_wkt_mask(self, wkt_list,target_mpp, contour_fill_distinct = False):
    
        scale, rescale = iu.scale_mpp(self.mpp, target_mpp)
    
        AnnPoly = []
        for wkt in wkt_list:
            polygon = loads(wkt)
            AnnPoly.append(polygon)
        mPoly = MultiPolygon(AnnPoly)
    
        AnnCnt = []
        for poly in mPoly.geoms:
            exterior_coords = np.array(poly.exterior.coords)
            contour = exterior_coords.reshape((-1, 1, 2)).astype(np.int32)
            AnnCnt.append((contour*scale).astype(int))
    
        wsi_dims = (int(self.pixel_dimensions[1]*scale), int(self.pixel_dimensions[0]*scale))
    
        AnnMask = np.zeros(wsi_dims, dtype=np.uint8)
    
        if contour_fill_distinct:
            for idx, Cnt in enumerate(AnnCnt):
                cv2.drawContours(AnnMask, [Cnt], -1, (idx+1), thickness=cv2.FILLED);
        else:
            for idx, Cnt in enumerate(AnnCnt):
                cv2.drawContours(AnnMask, [Cnt], -1, (1), thickness=cv2.FILLED);
        return AnnMask

    def get_pixeldim_at_mpp(self, target_mpp):
    
        scale, rescale = iu.scale_mpp(self.mpp, target_mpp)
        target_dims = (int(self.pixel_dimensions[1]*scale), int(self.pixel_dimensions[0]*scale))
        
        return target_dims

    def get_ann_region(self, ann, target_mpp):
        """
        Return np.array
        """

        scale_factor, rescale_factor = iu.scale_mpp(self.mpp, target_mpp)
    
        x = int(ann['minx'])
        y = int(ann['miny'])
        w = ann['w']
        h = ann['h']

        rescaled_dims =  (int(w*scale_factor), int(h*scale_factor))

        region  = np.array(core.get_region(self.slideRef,
                                           x=x, y=y,
                                           width=w, height=h,
                                           scale=1,
                                           sessionID=self.sessionID,
                                           format='jpg',
                                           quality=100)
                          )

        rescaled_region = cv2.resize(region, rescaled_dims)
 
        return rescaled_region

    def add_annotation(self, 
                       wkt, 
                       lineThickness= 2,
                       classification = '1AlgoG3', 
                       color = '#FF0000', 
                       layerID = 11, 
                       notes = None
                      ):
        """
        """
        
        ann = core.dummy_annotation()
        ann['geometry'] = wkt
        ann['lineThickness'] = lineThickness
        ann['color'] = color
        ann['classification'] = classification

        if notes is not None:
            ann['notes'] = notes
        else:
            ann['notes'] = ann['classification']
    
        
        core.add_annotation(self.slideRef, 
                            classification= ann['classification'], 
                            notes= ann['notes'], 
                            ann= ann, 
                            color= ann['color'], 
                            layerID= layerID, 
                            sessionID= self.sessionID
        )
        
    def delete_all_annotation(self):
        delete_bool =  input("This will delete all annotations from the slide, are you sure (Yes/No)?")
        if delete_bool == 'Yes':
            delete_bool = True
        else:
            delete_bool = False
            
        if delete_bool:
            core.clear_all_annotations(self.slideRef, sessionID=self.sessionID)
            print('All Annotations Cleared')
        else:
            print('Unsuccessful')

    def _delete_all_annotation(self):
        """
        Will Delete All Annotation without Warning. Use Carefully.
        """
        core.clear_all_annotations(self.slideRef, sessionID=self.sessionID)
        
    def delete_annotation_layer(self, layerID):
        core.clear_annotations(self.slideRef, layerID, sessionID=self.sessionID)


def get_pma_contour(geometry):
    contour_coordinates = geometry.split('POLYGON')[-1].split('(')[-1].split(')')[0].split(',')
    
    contour = []
    for coordinates in contour_coordinates:
        contour.append((float(coordinates.split()[0]),float(coordinates.split()[1])))
    return(contour)

def disconnect_pma_session():
    core.disconnect(sessionID=core.who_am_i()['sessionID'])
    return

class PMA_Slicer:
    """
    Input:
    PMA_Slide : PMA_Slide object 
    """
    def __init__(self,PMA_Slide):
        self.slide = PMA_Slide
    
    def get_scale_from_zoomlevel(self, zoom_level):
        """
        Input:
        zoom_level -  The zoom level at which to calculate the stats from
        slide -  A PMA_Slide object
        """
    
        mag_at_zoomlevel = self.slide.get_mag_at_zoomlevel(zoom_level)
        mpp_at_zoomlevel = self.slide.get_mpp_at_zoomlevel(zoom_level)

        scale_factor, rescale_factor = iu.scale_mpp(self.slide.mpp, mpp_at_zoomlevel)
    
        return(scale_factor, rescale_factor)
    
    def slice_annotation(self,annotation,patch_dims,overlap,target_mpp):

        scale_factor, rescale_factor = iu.scale_mpp(self.slide.mpp,target_mpp)
        
        geometry = annotation['Geometry']
        contour = get_pma_contour(geometry)
        minx, miny, maxx, maxy = Polygon(contour).bounds 
        
        start_x = int(minx)
        stop_x = int(maxx)

        start_y = int(miny)
        stop_y = int(maxy)

        x_patch, y_patch = patch_dims
        x_overlap, y_overlap = overlap
        
        step_x = x_patch-x_overlap
        step_y = y_patch-y_overlap
        
        x_patch_rescaled = int(x_patch*rescale_factor)
        y_patch_rescaled = int(y_patch*rescale_factor)

        x_overlap_rescaled = int(x_overlap*rescale_factor)
        y_overlap_rescaled = int(y_overlap*rescale_factor)

        step_x_rescaled = int(step_x*rescale_factor)
        step_y_rescaled = int(step_y*rescale_factor)
        
        min_slide_dim = min(self.slide.pixel_dimensions[0],self.slide.pixel_dimensions[1])
        
        if x_patch_rescaled> min_slide_dim or y_patch_rescaled > min_slide_dim:
            raise ValueError("Patch size greater than slide dimension") 

        #print(f"x_patch_rescaled:{x_patch_rescaled}, y_patch_rescaled:{y_patch_rescaled}")
        #print(f"x_overlap_rescaled:{x_overlap_rescaled}, y_overlap_scaled:{y_overlap_rescaled}")
        #print(f"step_x_rescaled:{step_x_rescaled}, step_y_rescaled:{step_y_rescaled}")
        
        patch_counter = 0
        for j in range(start_y, stop_y,step_y_rescaled):
            for i in range(start_x,stop_x,step_x_rescaled):

                if (i+step_x_rescaled)>stop_x:
                    i = stop_x-step_x_rescaled-x_overlap_rescaled 
        
                if (j+step_y_rescaled)>stop_y:
                    j = stop_y-step_y_rescaled-y_overlap_rescaled 
            
                patch_counter+=1
        
        patches = []
        coordinates = []

        pbar = tqdm(total=patch_counter, desc="Slicing Patches")
        
        for j in range(start_y, stop_y,step_y_rescaled):
            for i in range(start_x,stop_x,step_x_rescaled):
                
                if (i+step_x_rescaled)>stop_x:
                    i = stop_x-step_x_rescaled-x_overlap_rescaled 
        
                if (j+step_y_rescaled)>stop_y:
                    j = stop_y-step_y_rescaled-y_overlap_rescaled 
            
                image  = np.array(core.get_region(self.slide.slideRef,
                                                  x=i, y=j,
                                                  width=x_patch_rescaled, height=y_patch_rescaled,
                                                  scale=1,
                                                  sessionID=self.sessionID,
                                                  format='jpg',
                                                  quality=100)
                                 )
                resized_image = cv2.resize(image, (x_patch, y_patch))
                coordinates.append((i,j))
                #np_image = pm.np.array(image)
                patches.append(resized_image)
                #p.update()
                pbar.update(1)
        return(patches, coordinates)
    
    def slice_whole_slide(self,
                          patch_dims,
                          overlap_dims,
                          target_mpp,
                          start_x= None,
                          stop_x= None,
                          start_y= None,
                          stop_y = None
                         ):
        """
        Example Usage:
            zoom_level = 7.03
            scale_factor, rescale_factor = slicer.get_scale_from_zoomlevel(zoom_level)
            patch_dims = (2048,2048)
            overlap = (256,256)
        Return:
            patches, coordinates
        """
        scale_factor, rescale_factor = iu.scale_mpp(self.slide.mpp,target_mpp)
        
        if start_x == None or stop_x == None or start_y == None or stop_y == None:
            print('Slicing Whole slide')
            start_x = int(0)
            stop_x =  int(self.slide.pixel_dimensions[0])

            start_y = int(0)
            stop_y =  int(self.slide.pixel_dimensions[1])
        else:
            start_x = int(start_x)
            stop_x =  int(stop_x)

            start_y = int(start_y)
            stop_y =  int(stop_y)
            
        x_patch, y_patch = patch_dims
        x_overlap, y_overlap = overlap_dims
        
        step_x = x_patch-x_overlap
        step_y = y_patch-y_overlap
        
        x_patch_rescaled = int(x_patch*rescale_factor)
        y_patch_rescaled = int(y_patch*rescale_factor)

        x_overlap_rescaled = int(x_overlap*rescale_factor)
        y_overlap_rescaled = int(y_overlap*rescale_factor)

        step_x_rescaled = int(step_x*rescale_factor)
        step_y_rescaled = int(step_y*rescale_factor)
        
        min_slide_dim = min(self.slide.pixel_dimensions[0],self.slide.pixel_dimensions[1])
        
        if x_patch_rescaled> min_slide_dim or y_patch_rescaled > min_slide_dim:
            raise ValueError("Patch size greater than slide dimension")           

        
        patch_counter = 0
        for j in range(start_y, stop_y,step_y_rescaled):
            for i in range(start_x,stop_x,step_x_rescaled):

                if (i+step_x_rescaled)>stop_x:
                    #i = stop_x-step_x_rescaled-x_overlap_rescaled 
                    i = stop_x-step_x_rescaled
        
                if (j+step_y_rescaled)>stop_y:
                    #j = stop_y-step_y_rescaled-y_overlap_rescaled
                    j = stop_y-step_y_rescaled 
            
                patch_counter+=1
        
        patches = []
        coordinates = []

        pbar = tqdm(total=patch_counter, desc="Slicing Patches")
        for j in range(start_y, stop_y,step_y_rescaled):
            for i in range(start_x,stop_x,step_x_rescaled):
                
                if (i+step_x_rescaled)>stop_x:
                    #i = stop_x-step_x_rescaled-x_overlap_rescaled
                    i = stop_x-step_x_rescaled
        
                if (j+step_y_rescaled)>stop_y:
                    #j = stop_y-step_y_rescaled-y_overlap_rescaled
                    j = stop_y-step_y_rescaled 
            
                image  = np.array(core.get_region(self.slide.slideRef,
                                                  x=i, y=j,
                                                  width=x_patch_rescaled, height=y_patch_rescaled,
                                                  scale=scale_factor,
                                                  sessionID=self.sessionID,
                                                  format='jpg',
                                                  quality=100)
                                 )

                resized_image = cv2.resize(image, (x_patch, y_patch))
                
                coordinates.append((i,j))
                patches.append(resized_image)
                pbar.update(1)

        return(patches, coordinates)
    

def create_slideRef(slide_name):
    """
    Input: 
    slide_name -> Slide name without the format.
    """
    slideRef = f"CAIB_WSI/{slide_name[:4]}/{slide_name[:16]}/{slide_name[:21]}/{slide_name[:28]}/{slide_name}.svs" 
    return(slideRef)

def create_ann_mask(AnnWkt, Image_Shape, scale_factor = 1):
    """
    Input:
    AnnWkt -> List of wkt's for mask
    Image_Shape -> Mask Dimensions
    """
    
    AnnPoly = []
    for wkt in AnnWkt:
        polygon = loads(wkt)
        AnnPoly.append(polygon)
    mPoly = MultiPolygon(AnnPoly)
    
    AnnCnt = []
    for poly in mPoly.geoms:
        exterior_coords = np.array(poly.exterior.coords)
        contour = exterior_coords.reshape((-1, 1, 2)).astype(np.int32)
        AnnCnt.append((contour*scale_factor).astype(int))

    AnnMask = np.zeros(Image_Shape, dtype=np.uint8)
    
    for idx, Cnt in enumerate(AnnCnt):
        cv2.drawContours(AnnMask, [Cnt], -1, (idx+1), thickness=cv2.FILLED);
    
    return(AnnMask)

