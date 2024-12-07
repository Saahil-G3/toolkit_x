import cv2
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from .colors import get_rgb_colors


def plot_image(
    image, 
    plot=True,
    alpha=150,
    figsize=(5, 5),
    axis="off",
    title=None,
    title_fontsize=14,
    dpi=300,
    save_path=False
    ):
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
    plt.axis(axis)
    if title:
        plt.title(title, fontsize=title_fontsize)
    if save_path:
        plt.savefig(
            save_path, bbox_inches="tight", pad_inches=0, transparent=True, dpi=dpi
        )
        print("Image saved")
    if plot:
        plt.show()
    else:
        plt.close()

        
def plot_overlay(
    image,
    mask,
    plot=True,
    alpha=150,
    figsize=(5, 5),
    axis="off",
    title=None,
    title_fontsize=14,
    dpi=300,
    save_path=False,
):
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
    plt.imshow(get_overlay(image, mask, alpha=alpha))
    plt.axis(axis)
    if title:
        plt.title(title, fontsize=title_fontsize)
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
    images, masks, title=None, save_path=False, figsize=(15, 5), plot=True, alpha=150
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
        ax.imshow(get_overlay(images[i], masks[i], alpha=alpha))
        ax.axis("off")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=500)

    if plot:
        plt.show()
    else:
        plt.close(fig)


def get_overlay(image, mask, alpha=200):

    # Generate random colors for each class
    classes = np.unique(mask)
    colors = get_rgb_colors(len(classes))

    # Create an overlay image and alpha channel
    overlayed_image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    alpha_channel = np.zeros(mask.shape, dtype=np.uint8)

    # Apply colors and alpha values for each class (except background)
    for idx, class_id in enumerate(classes[classes != 0]):
        overlayed_pixels = mask == class_id
        overlayed_image[overlayed_pixels] = colors[idx]
        alpha_channel[overlayed_pixels] = alpha

    # Merge overlay and alpha channel
    overlayed_image = cv2.merge((*cv2.split(overlayed_image), alpha_channel))

    return overlayed_image