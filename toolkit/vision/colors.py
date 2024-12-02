import re
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from mycolorpy import colorlist as mcp

color_definitions = [
    {"name": "Aquamarine", "rgb": (128, 255, 219), "hex": "#80ffdb"},
    {"name": "Lime Green", "rgb": (175, 252, 65), "hex": "#affc41"},
    {"name": "Magenta", "rgb": (255, 0, 110), "hex": "#ff006e"},
    {"name": "Peachy Orange", "rgb": (249, 132, 74), "hex": "#f9844a"},
    {"name": "Soft Yellow", "rgb": (255, 209, 102), "hex": "#ffd166"},
    {"name": "Forest Green", "rgb": (51, 61, 41), "hex": "#333d29"},
    {"name": "Deep Teal", "rgb": (7, 59, 76), "hex": "#073b4c"},
    {"name": "Goldenrod", "rgb": (255, 190, 11), "hex": "#ffbe0b"},
    {"name": "Pumpkin Orange", "rgb": (251, 86, 7), "hex": "#fb5607"},
    {"name": "Amethyst Purple", "rgb": (131, 56, 236), "hex": "#8338ec"},
    {"name": "Sky Blue", "rgb": (58, 134, 255), "hex": "#3a86ff"},
    {"name": "Crimson Red", "rgb": (249, 65, 68), "hex": "#f94144"},
    {"name": "Tangerine", "rgb": (243, 114, 44), "hex": "#f3722c"},
    {"name": "Watermelon Pink", "rgb": (239, 71, 111), "hex": "#ef476f"},
    {"name": "Sunset Orange", "rgb": (248, 150, 30), "hex": "#f8961e"},
    {"name": "Mint Green", "rgb": (6, 214, 160), "hex": "#06d6a0"},
    {"name": "Teal Gray", "rgb": (77, 144, 142), "hex": "#4d908e"},
    {"name": "Honey Yellow", "rgb": (249, 199, 79), "hex": "#f9c74f"},
    {"name": "Olive Green", "rgb": (144, 190, 109), "hex": "#90be6d"},
    {"name": "Seafoam Green", "rgb": (67, 170, 139), "hex": "#43aa8b"},
    {"name": "Violet", "rgb": (105, 48, 195), "hex": "#6930c3"},
    {"name": "Slate Gray", "rgb": (87, 117, 144), "hex": "#577590"},
    {"name": "Ocean Blue", "rgb": (39, 125, 161), "hex": "#277da1"},
    {"name": "Royal Purple", "rgb": (116, 0, 184), "hex": "#7400b8"},
    {"name": "Sandy Brown", "rgb": (166, 138, 100), "hex": "#a68a64"},
    {"name": "Olive Drab", "rgb": (65, 72, 51), "hex": "#414833"},
    {"name": "Plum", "rgb": (60, 22, 66), "hex": "#3c1642"},
    {"name": "Deep Cyan", "rgb": (8, 99, 117), "hex": "#086375"},
    {"name": "Turquoise", "rgb": (29, 211, 176), "hex": "#1dd3b0"},
    {"name": "Pale Lime", "rgb": (178, 255, 158), "hex": "#b2ff9e"},
    {"name": "Azure", "rgb": (17, 138, 178), "hex": "#118ab2"},
]


def percentage_to_hex_alpha(percentage):
    """
    Convert an opacity percentage (0-100) to a hexadecimal alpha value.

    Args:
    - percentage (int or float): Opacity percentage (0 to 100).

    Returns:
    - str: Hexadecimal alpha value (2 digits).
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100.")
    alpha_decimal = round(percentage / 100 * 255)
    return format(alpha_decimal, "02X")


def plot_predefined_colors():
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in enumerate(color_definitions):
        rgb = [x / 255 for x in color["rgb"]]  # Normalize RGB values
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=rgb))
        ax.text(1.1, i + 0.5, f'{i}: {color["name"]}', va="center", fontsize=10)

    # Aesthetics
    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(color_definitions))
    ax.axis("off")
    plt.title("Color Definitions", fontsize=10)
    # plt.tight_layout()
    plt.show()


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
        "Blues",
        "Greens",
        "Reds",
        "Purples",
        "Oranges",
        # Diverging
        "Spectral",
        "coolwarm",
        "RdBu",
        "PiYG",
        "BrBG",
        # Perceptually Uniform
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        # Cyclic
        "twilight",
        "hsv",
        # Miscellaneous
        "rainbow",
        "terrain",
        "ocean",
        "cubehelix",
    ]

    if 0 <= index < len(colormaps):
        return colormaps[index]
    else:
        raise ValueError("Index out of range. Please select an index between 0 and 20.")


def get_hex_cmap_range(n, cmap="viridis"):
    hex_cmap_range = mcp.gen_color(cmap=cmap, n=n)
    return hex_cmap_range


def get_rgb_cmap_range(n, cmap="viridis"):
    hex_cmap_range = mcp.gen_color(cmap=cmap, n=n)
    rgb_cmap_range = [hex_to_rgb(color) for color in cmap_range]
    return rgb_cmap_range


def get_hex_colors(n, cmap="Spectral"):
    if n <= len(color_definitions):
        rgb_colors = [color_dict["hex"] for color_dict in color_definitions[:n]]
    else:
        rgb_colors1 = [
            color_dict["hex"]
            for color_dict in color_definitions[: len(color_definitions)]
        ]
        rgb_colors2 = mcp.gen_color(cmap=cmap, n=n - len(color_definitions))
        rgb_colors = rgb_colors1 + rgb_colors2

    return rgb_colors


def get_rgb_colors(n, cmap="Spectral"):
    if n <= len(color_definitions):
        rgb_colors = [color_dict["rgb"] for color_dict in color_definitions[:n]]
    else:
        rgb_colors1 = [
            color_dict["rgb"]
            for color_dict in color_definitions[: len(color_definitions)]
        ]
        colors2 = mcp.gen_color(cmap=cmap, n=n - len(color_definitions))
        rgb_colors2 = [hex_to_rgb(color) for color in colors2]
        rgb_colors = rgb_colors1 + rgb_colors2

    return rgb_colors


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    if len(hex_color) not in {6, 8} or not re.match(r"^[0-9a-fA-F]{6}$", hex_color):
        raise ValueError(f"Invalid hex color string: {hex_color}")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
