import os
import copy
import numpy as np
import random
import cv2
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from tifffile import imread, imsave
import warnings
from scipy.ndimage import binary_erosion
from mycolorpy import colorlist as mcp

PanNuke_Label_Map = {
    0: "BACKGROUND",
    1: "Neoplastic",
    2: "Inflammatory",
    3: "Connective",
    4: "Dead",
    5: "Epithelial",
}

def square_contour(x, y, patch_size):
    # Define the corners of the square
    X = [x, x + patch_size, x + patch_size, x, x]
    Y = [y, y, y - patch_size, y - patch_size, y]
    
    # Convert to contour format
    return convert_geojson_contour(X, Y)

def hex_to_rgb(hex_color):
    # Ensure the input is in the format #RRGGBB
    if hex_color.startswith('#'):
        hex_color = hex_color[1:]
    
    # Convert hex to RGB
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    return (r, g, b)

def get_cmap(index):
    """
    This function returns a colormap based on the given index. The colormaps are divided into categories:
    
    1. Sequential: Smooth transition through shades of a single color.
       - Index 0-4: Blues, Greens, Reds, Purples, Oranges
       
    2. Diverging: Move from one color to another, useful for showing deviation from a center.
       - Index 5-9: Spectral, coolwarm, RdBu, PiYG, BrBG
       
    3. Perceptually Uniform: Designed to appear uniformly spaced, even for people with color vision deficiencies.
       - Index 10-14: viridis, plasma, inferno, magma, cividis
       
    4. Cyclic: Colormaps that wrap around, useful for cyclic data (e.g., phase, angle).
       - Index 15-16: twilight, hsv
       
    5. Miscellaneous: Other colormaps representing natural phenomena or broader ranges.
       - Index 17-20: rainbow, terrain, ocean, cubehelix
    """
    colormaps = [
        # Sequential
        "Blues", "Greens", "Reds", "Purples", "Oranges",
        # Diverging
        "Spectral", "coolwarm", "RdBu", "PiYG", "BrBG",
        # Perceptually Uniform
        "viridis", "plasma", "inferno", "magma", "cividis",
        # Cyclic
        "twilight", "hsv",
        # Miscellaneous
        "rainbow", "terrain", "ocean", "cubehelix"
    ]

    if 0 <= index < len(colormaps):
        return colormaps[index]
    else:
        raise ValueError("Index out of range. Please select an index between 0 and 20.")

def get_rgb_colors(n, cmap='Spectral_r'):
    '''
    cmap options
    Perceptually Uniform Colormaps:
        'viridis': A popular colormap with green, blue, and yellow.
        'plasma': A high-contrast colormap with purples, reds, and yellow.
        'inferno': Black to yellow, through reds and purples.
        'magma': Dark purple to yellowish tones.
    '''
    clrs=mcp.gen_color(cmap=cmap,n=n)
    color_list=[hex_to_rgb(clr) for clr in clrs]
    return color_list

def convert_geojson_contour(X,Y):
    cnt_list = []
    for x,y in zip(X,Y):
        cnt_list.append([x,y]) 
    
    return(np.array(cnt_list).astype(int))

def get_wkt(X,Y):
    wkt = str()

    for x,y in zip(X,Y):
        wkt = wkt + f"{int(x)} {int(y)},"

    wkt = wkt + f"{int(X[0])} {int(Y[0])},"
    wkt = wkt[:-1]
    wkt =  f"POLYGON (( {wkt} ))"
    
    return(wkt)


def plot_image_series(images, title= None, save_path= False, figsize =(15, 5), plot=True):
    num_images = len(images)
    
    if num_images < 1:
        print("No images to plot.")
        return

    # Create a grid of subplots
    rows = 1
    cols = num_images
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_images == 1:
        axes = [axes]  # Ensure it's a list even if there's only one image

    for i, ax in enumerate(axes):
        if title == None:
            pass
        else:
            ax.set_title(title[i])
        ax.imshow(images[i])
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi = 300)

    if plot:
        plt.show()
    else:
        plt.close(fig) 
        
def plot_overlay_series(images, masks, title = None, save_path = False, figsize =(15, 5), plot=True):
    num_images = len(images)
    
    if num_images < 1:
        print("No images to plot.")
        return

    # Create a grid of subplots
    rows = 1
    cols = num_images
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if num_images == 1:
        axes = [axes]  # Ensure it's a list even if there's only one image

    for i, ax in enumerate(axes):
        if title == None:
            pass
        else:
            ax.set_title(title[i])
        ax.imshow(images[i])
        ax.imshow(get_random_overlay(images[i],masks[i]))
        ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi = 500)
    
    if plot:
        plt.show()
    else:
        plt.close(fig)

def get_color_dict():
    """
    """
    color_dict = {
        0: {'RGB': (255,0,0),    'color': 'Red',    'HEX': '#FF0000'},
        1: {'RGB': (0, 0, 0),    'color': 'Black',  'HEX': '#000000'},
        2: {'RGB': (0, 255, 0),  'color': 'Green',  'HEX': '#00FF00'},
        3: {'RGB': (0, 0, 255),  'color': 'Blue',   'HEX': '#0000FF'},
        4: {'RGB': (255,255,0),  'color': 'Yellow', 'HEX': '#FFFF00'},
        5: {'RGB': (255,255,255),'color': 'White',  'HEX': '#FFFFFF'},
        6: {'RGB': (128,0,128),  'color': 'Purple', 'HEX': '#800080'},
        7: {'RGB': (0,255,255),  'color': 'Cyan',   'HEX': '#00FFFF'},
        8: {'RGB': (255,0,255),  'color': 'Magenta','HEX': '#FF00FF'},
    }
    
    return(color_dict)

def get_color_dict_geojson():
    
    color_dict = {
        0: {'RGB': [255,0,0],     'color': 'Red'},
        1: {'RGB': [0, 0, 0],     'color': 'Black'},
        2: {'RGB': [0, 255, 0],   'color': 'Green'},
        3: {'RGB': [0, 0, 255],   'color': 'Blue'  },
        4: {'RGB': [255,255,0],   'color': 'Yellow'},
        5: {'RGB': [255,255,255], 'color': 'White'},
        6: {'RGB': [128,0,128],   'color': 'Purple'},
        7: {'RGB': [0,255,255],   'color': 'Cyan'},
        8: {'RGB': [255,0,255],   'color': 'Magenta'},
    }
    return(color_dict)

def plot_overlay(image, mask, save_path = False,figsize=(10,10), dpi=300, plot=True):
    """
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.imshow(get_random_overlay(image,mask,alpha = 120))
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi = dpi)
        print('Image saved')
    
    if plot:
        plt.show()
    else:
        plt.close()

def plot_image(image, figsize=(5,5), save_path = False, plot=True):
    """
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, transparent=True, dpi = 300)
        print('Image saved')
    
    if plot:
        plt.show()
    else:
        plt.close()

def scale_mpp(current_mpp,target_mpp):
    """
    """
    scale_factor =  current_mpp/target_mpp
    rescale_factor = 1/scale_factor
    
    return scale_factor,rescale_factor

def get_random_overlay(image,mask,alpha = 150):
    
    class_idx = np.unique(mask)
    overlayed_image = np.zeros_like(image, dtype=np.uint8)
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_value_for_cells = alpha

    colors=get_rgb_colors(len(class_idx))
    
    for idx,  i in enumerate(class_idx):
        if i != 0:
            overlayed_pixels = mask == i
            overlayed_image[overlayed_pixels] = colors[idx]#(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
            alpha_channel[overlayed_pixels] = alpha_value_for_cells
    alpha_channel[mask == 0] = 0
    overlayed_image = cv2.merge((overlayed_image, alpha_channel))
    
    return overlayed_image

def get_classification_overlay(image,mask,alpha = 150):
    
    class_idx = np.unique(mask)
    colors=get_rgb_colors(len(class_idx))
    
    if len(class_idx) > len(color_dict):
        print("Classess more than colors")
        return
    
    overlayed_image = np.zeros_like(image, dtype=np.uint8)
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_value_for_cells = alpha
    
    print(f'Total classes: {len(class_idx)} --> {class_idx}')
    
    for i in tqdm(class_idx):
        if i != 0:
            overlayed_pixels = mask == i
            overlayed_image[overlayed_pixels] = color_dict[i]
            alpha_channel[overlayed_pixels] = alpha_value_for_cells
    alpha_channel[mask == 0] = 0
    overlayed_image = cv2.merge((overlayed_image, alpha_channel))
    
    return overlayed_image

def contour_to_array(contour,patch_height, patch_width,fill_number=1):
    """
    """
    from skimage.draw import polygon
    seg_mask = np.zeros((patch_height, patch_width))
    rr, cc = polygon(contour[:, 1], contour[:, 0])
    seg_mask[rr, cc] = fill_number
    
    return seg_mask

def get_bounding_box_cords(single_channel_mask):
    """
    """
    nonzero_indices = np.nonzero(single_channel_mask)
    min_row, min_col = np.min(nonzero_indices[0]), np.min(nonzero_indices[1])
    max_row, max_col = np.max(nonzero_indices[0]), np.max(nonzero_indices[1])
    
    return [min_row, min_col, max_row, max_col]

def is_array_touching_boundary(obj_mask):
    """
    Check if the object touches the boundary along any axis
    
    Input:
    obj_mask: np.array mask containing the contour of a single object
    """
    return np.any(obj_mask[0, :]) or np.any(obj_mask[-1, :]) or np.any(obj_mask[:, 0]) or np.any(obj_mask[:, -1])

def show_image_big(img, **kwargs):
    """Plot large image at different resolutions."""
    fig, ax = plt.subplots(2,4, figsize=(16,8))
    mid = [s//2 + 600 for s in img.shape[:2]]
    for a,t,u in zip(ax.ravel(),[1,2,4,8,16,32,64,128],[16,8,4,2,1,1,1,1]):
        sl = tuple(slice(c - s//t//2, c + s//t//2, u) for s,c in zip(img.shape[:2],mid))
        a.imshow(img[sl], **kwargs)
        a.axis('off')
    plt.tight_layout()
    plt.show()
    
def extract_patches_with_coordinates(image, patch_dims, overlap_dims):
    """
    Extract patches from a large numpy array with specified patch size and overlap.

    Parameters:
    - image: numpy array, the input image
    - patch_size: tuple, (h, w), size of the patches to be extracted
    - overlap: tuple, (oh, ow), overlap in the vertical and horizontal directions

    Returns:
    - patches: list, a list of extracted patches
    - coordinates: list, a list of tuples containing the (start_row, start_col) coordinates of each patch
    """

    h, w = patch_dims
    oh, ow = overlap_dims

    patches = []
    coordinates = []
    
    flag = 0
    
    print("Extracting Patches")
    for i in range(0, image.shape[0], h - oh):
        if i+h>image.shape[0]:
            i=image.shape[0]-h
            
        for j in range(0, image.shape[1], w - ow):
            if j+w>image.shape[1]:
                j=image.shape[1]-w
    
            patch = image[i:i+h, j:j+w,:]

            patches.append(patch)
            coordinates.append((i, j))
            flag +=1
    
    print(f"Complete!")
    print(f"Extracted {flag} patches.")
    
    return np.array(patches), coordinates
    
