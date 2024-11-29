import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .colors import get_rgb_colors

def plot_image(image, figsize=(5, 5), save_path=False, plot=True):
    """
    Plot an image with optional saving and display options.

    Args:
    - image: The image to be plotted, as a numpy array or similar.
    - figsize: Tuple specifying the figure size. Default is (5, 5).
    - save_path: If specified, saves the image to the provided file path. Default is False.
    - plot: If True, displays the image. If False, the figure is closed. Default is True.
    """

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")

    if save_path:
        plt.savefig(
            save_path, bbox_inches="tight", pad_inches=0, transparent=True, dpi=300
        )
        print("Image saved")

    if plot:
        plt.show()
    else:
        plt.close()

def plot_overlay(image, mask, save_path=False, figsize=(10, 10), dpi=300, plot=True):
    """
    Overlay an image with a mask using a random color scheme.
    Args:
    - image (numpy.ndarray): Base image.
    - mask (numpy.ndarray): Mask to overlay on the image.
    - alpha (int): Transparency value for the overlay.
    Returns:
    - numpy.ndarray: Overlayed image with alpha channel.
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.imshow(get_random_overlay(image, mask, alpha=120))
    plt.axis("off")

    if save_path:
        plt.savefig(
            save_path, bbox_inches="tight", pad_inches=0, transparent=True, dpi=dpi
        )
        print("Image saved")

    if plot:
        plt.show()
    else:
        plt.close()

def plot_image_series(images, title=None, save_path=False, figsize=(15, 5), plot=True):
    """
    Plot a series of images with optional titles and save the plot to a file.
    Args:
    - images (list): List of images to plot.
    - title (list or None): List of titles for the images or None.
    - save_path (str or bool): Path to save the plot or False.
    - figsize (tuple): Size of the figure.
    - plot (bool): Whether to display the plot.
    """
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
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)

    if plot:
        plt.show()
    else:
        plt.close(fig)

def plot_overlay_series(
    images, masks, title=None, save_path=False, figsize=(15, 5), plot=True
):
    """
    Overlay a series of images with masks and plot them with optional titles.
    Args:
    - images (list): List of base images.
    - masks (list): List of masks to overlay on the images.
    - title (list or None): List of titles for the images or None.
    - save_path (str or bool): Path to save the plot or False.
    - figsize (tuple): Size of the figure.
    - plot (bool): Whether to display the plot.
    """
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
        ax.imshow(get_random_overlay(images[i], masks[i]))
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if plot:
        plt.show()
    else:
        plt.close(fig)


def get_random_overlay(image, mask, alpha=150):
    """
    Generate a random overlay on an image based on a segmentation mask.

    Args:
    - image: The input image to overlay with masks.
    - mask: The segmentation mask where each unique value represents a different class.
    - alpha: Integer specifying the transparency level for overlayed areas. Default is 150.

    Returns:
    - overlayed_image: Image with an alpha channel containing the overlaid masks.
    """

    class_idx = np.unique(mask)
    overlayed_image = np.zeros_like(image, dtype=np.uint8)
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_value_for_cells = alpha

    colors = get_rgb_colors(len(class_idx))

    for idx, i in enumerate(class_idx):
        if i != 0:
            overlayed_pixels = mask == i
            overlayed_image[overlayed_pixels] = colors[
                idx
            ]  # (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
            alpha_channel[overlayed_pixels] = alpha_value_for_cells
    alpha_channel[mask == 0] = 0
    overlayed_image = cv2.merge((overlayed_image, alpha_channel))

    return overlayed_image


def get_classification_overlay(image, mask, alpha=150):
    """
    Create a classification overlay on an image based on a mask.

    Args:
    - image: The input image to overlay.
    - mask: The segmentation mask where each unique value represents a different class.
    - alpha: Integer specifying the transparency level for overlayed areas. Default is 150.

    Returns:
    - overlayed_image: Image with an alpha channel containing the overlaid classification masks.
    """

    class_idx = np.unique(mask)
    colors = get_rgb_colors(len(class_idx))

    if len(class_idx) > len(color_dict):
        print("Classess more than colors")
        return

    overlayed_image = np.zeros_like(image, dtype=np.uint8)
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_value_for_cells = alpha

    print(f"Total classes: {len(class_idx)} --> {class_idx}")

    for i in tqdm(class_idx):
        if i != 0:
            overlayed_pixels = mask == i
            overlayed_image[overlayed_pixels] = color_dict[i]
            alpha_channel[overlayed_pixels] = alpha_value_for_cells
    alpha_channel[mask == 0] = 0
    overlayed_image = cv2.merge((overlayed_image, alpha_channel))

    return overlayed_image