import cv2
import math
import torch
import geojson
import numpy as np
from shapely.wkt import loads
from shapely.geometry import mapping
from shapely.geometry import box as Box
from shapely.validation import make_valid
from shapely.geometry import (
    Point,
    Polygon,
    LineString,
    LinearRing,
    MultiPolygon,
    MultiLineString,
    MultiPoint,
    GeometryCollection,
)

is_collection = {}
is_collection["Point"] = False
is_collection["Polygon"] = False
is_collection["LineString"] = False
is_collection["LinearRing"] = False
is_collection["MultiPoint"] = True
is_collection["MultiPolygon"] = True
is_collection["MultiLineString"] = True
is_collection["GeometryCollection"] = True

from shapely.prepared import prep as prep_geom_for_query

# from shapely.geometry import shape as Shape
# from shapely.ops import unary_union
# from shapely import wkt

# from shapely.strtree import STRtree

class GeomStats():
    def __init__(self):
        pass

    def get_circularity(self, geom):
        area = geom.area
        perimeter = geom.length
    
        if perimeter == 0:
            return 0.0
    
        circularity = (4 * math.pi * area) / (perimeter ** 2)
    
        return circularity

    def get_major_minor_axes(self, geom):
        """
        Calculates the major and minor axes of a geometry's minimum rotated rectangle.
    
        This function computes the major and minor axes of the bounding box that minimally encloses the input Shapely geometry, while allowing rotation. The axes are derived from the dimensions of this rotated rectangle.
    
        Args:
            geom: A Shapely geometry object for which the major and minor axes are to be calculated.
    
        Returns:
            dict: A dictionary containing:
                - "minor_axis": Length of the shorter side of the minimum rotated rectangle.
                - "major_axis": Length of the longer side of the minimum rotated rectangle.
        """
        # min_rot_rect = geom.minimum_rotated_rectangle
        # min_rot_rect_coords = list(min_rot_rect.exterior.coords)
        # x_coords = [coord[0] for coord in min_rot_rect_coords]
        # y_coords = [coord[1] for coord in min_rot_rect_coords]
        # width = max(x_coords) - min(x_coords)
        # height = max(y_coords) - min(y_coords)
    
        # major_minor_axis_dict = {}
        # major_minor_axis_dict["minor_axis"] = min(width, height)
        # major_minor_axis_dict["major_axis"] = max(width, height)

        # Get the minimum rotated rectangle
        min_rot_rect = geom.minimum_rotated_rectangle
        
        # Extract the coordinates of the rectangle's exterior
        min_rot_rect_coords = list(min_rot_rect.exterior.coords)
        
        # Calculate the lengths of the sides of the rectangle
        side_lengths = []
        for i in range(len(min_rot_rect_coords) - 1):
            x1, y1 = min_rot_rect_coords[i]
            x2, y2 = min_rot_rect_coords[i + 1]
            length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            side_lengths.append(length)
        
        # The major axis is the longest side, and the minor axis is the shortest side
        major_axis = max(side_lengths)
        minor_axis = min(side_lengths)
        
        return {"minor_axis": minor_axis, "major_axis": major_axis}
        #return major_minor_axis_dict

def get_major_minor_axes(geom):
    """
    Calculates the major and minor axes of a geometry's minimum rotated rectangle.

    This function computes the major and minor axes of the bounding box that minimally encloses the input Shapely geometry, while allowing rotation. The axes are derived from the dimensions of this rotated rectangle.

    Args:
        geom: A Shapely geometry object for which the major and minor axes are to be calculated.

    Returns:
        dict: A dictionary containing:
            - "minor_axis": Length of the shorter side of the minimum rotated rectangle.
            - "major_axis": Length of the longer side of the minimum rotated rectangle.
    """
    min_rot_rect = geom.minimum_rotated_rectangle
    min_rot_rect_coords = list(min_rot_rect.exterior.coords)
    x_coords = [coord[0] for coord in min_rot_rect_coords]
    y_coords = [coord[1] for coord in min_rot_rect_coords]
    width = max(x_coords) - min(x_coords)
    height = max(y_coords) - min(y_coords)

    major_minor_axis_dict = {}
    major_minor_axis_dict["minor_axis"] = min(width, height)
    major_minor_axis_dict["major_axis"] = max(width, height)
    return major_minor_axis_dict


def get_numpy_mask_from_geom(
    geom,
    mask_dims: tuple = None,
    origin: tuple = (0, 0),
    scale_factor: float = 1,
):
    """
    Generates a numpy mask from a Shapely geometry.

    This function creates a binary mask (numpy array) based on a given Shapely geometry. The geometry can consist of polygons with possible holes. If multiple geometries are detected, a warning is issued. The mask dimensions, origin, and scaling can be customized.

    Args:
        geom: A Shapely geometry object representing the shape to mask.
        mask_dims (tuple, optional): Dimensions of the output mask as (height, width).
            If not provided, dimensions are inferred from the bounding box of the geometry.
        origin (tuple, optional): The (x, y) origin of the mask. Default is (0, 0).
        scale_factor (float, optional): Scaling factor for the geometry coordinates.
            Default is 1.

    Returns:
        numpy.ndarray: A binary mask of shape `mask_dims` where the geometry is filled with 1s
        and holes are filled with 0s.

    Raises:
        Warning: If multiple geometries are detected in the input geometry.

    """
    if not mask_dims:
        minx, miny, maxx, maxy = geom.bounds
        width = int((maxx - minx)*scale_factor)
        height = int((maxy - miny)*scale_factor)
        mask_dims = (height, width)
        origin = (minx, miny)

    geom_dict = flatten_geom_collection(geom)
    if len(geom_dict) > 1:
        warnings.warn(
            f"Multiple geometries detected in tissue mask. Check: {', '.join(geom_dict.keys())}"
        )
    exterior, holes = [], []
    for polygon in geom_dict["Polygon"]:
        polygon_coordinates = get_polygon_coordinates_cpu(
            polygon, scale_factor=scale_factor, origin=origin
        )
        exterior.extend(polygon_coordinates[0])
        holes.extend(polygon_coordinates[1])
    mask = np.zeros(mask_dims, dtype=np.uint8)
    cv2.fillPoly(mask, exterior, 1)
    if len(holes) > 0:
        cv2.fillPoly(mask, holes, 0)
    return mask


def geom_to_geojson(geom):
    """
    Converts a Shapely geometry object to a GeoJSON feature.

    Args:
        geom (shapely.geometry.base.BaseGeometry): A Shapely geometry object.

    Returns:
        geojson.Feature: A GeoJSON feature representation of the input geometry.
    """
    geojson_feature = geojson.Feature(geometry=mapping(geom))
    return geojson_feature
    
def get_box(x, y, height, width):
    """
    Creates a rectangular bounding box geometry.

    Args:
        x (float): The x-coordinate of the box's lower-left corner.
        y (float): The y-coordinate of the box's lower-left corner.
        height (float): The height of the box.
        width (float): The width of the box.

    Returns:
        shapely.geometry.Polygon: A rectangular bounding box as a Shapely Polygon.
    """
    return Box(x, y, x + height, y + width)


def flatten_geom_collection(geom):
    """
    Flattens a geometry collection into a dictionary of geometry types.

    Args:
        geom (shapely.geometry.base.BaseGeometry): A Shapely GeometryCollection or any geometry.

    Returns:
        dict: A dictionary where keys are geometry types (e.g., 'Point', 'Polygon') and values are lists of geometries of that type.
    """
    geom_dict = {}
    stack = [geom]

    while stack:
        current_geom = stack.pop()
        geom_type = current_geom.geom_type

        if is_collection.get(geom_type, False):
            stack.extend(current_geom.geoms)
        else:
            geom_dict.setdefault(geom_type, []).append(current_geom)

    return geom_dict


def get_polygon_coordinates_cpu(polygon, scale_factor=1, origin=None):
    """
    Extracts exterior and interior (hole) coordinates of a polygon on the CPU.

    Args:
        polygon (shapely.geometry.Polygon): The polygon whose coordinates are to be extracted.
        scale_factor (float, optional): A scaling factor to apply to the coordinates. Default is 1.
        origin (numpy.ndarray or None, optional): The origin for translation. If None, uses [0, 0].

    Returns:
        tuple: A tuple containing:
            - list[numpy.ndarray]: The exterior coordinates of the polygon.
            - list[numpy.ndarray]: The interior (hole) coordinates of the polygon.
    """
    if origin is None:
        origin = np.zeros(2, dtype=np.float32)

    # Precompute scaling factor
    scale_factor = np.float32(scale_factor)

    # Exterior coordinates
    exterior = np.array(polygon.exterior.coords, dtype=np.float32)
    np.subtract(exterior, origin, out=exterior)  # In-place origin adjustment
    np.multiply(exterior, scale_factor, out=exterior)  # In-place scaling
    exterior = np.round(exterior).astype(np.int32)

    # Interior coordinates (holes)
    holes = [
        np.round(
            (np.array(interior.coords, dtype=np.float32) - origin) * scale_factor
        ).astype(np.int32)
        for interior in polygon.interiors
    ]

    return [exterior], holes


def get_polygon_coordinates_gpu(polygon, device, scale_factor=1, origin=None):
    """
    Extracts exterior and interior (hole) coordinates of a polygon on the GPU.

    Args:
        polygon (shapely.geometry.Polygon): The polygon whose coordinates are to be extracted.
        device (torch.device): The device (GPU) to perform the computation on.
        scale_factor (float, optional): A scaling factor to apply to the coordinates. Default is 1.
        origin (torch.Tensor or None, optional): The origin for translation. If None, uses a tensor of zeros.

    Returns:
        tuple: A tuple containing:
            - list[torch.Tensor]: The exterior coordinates of the polygon.
            - list[torch.Tensor]: The interior (hole) coordinates of the polygon.
    """
    if origin is None:
        origin = torch.tensor([0, 0], dtype=torch.float32, device=device)
    exterior = (
        torch.tensor(polygon.exterior.coords, dtype=torch.float32, device=device)
        - origin
    ) * scale_factor
    exterior = exterior.round().to(torch.int32)

    holes = [
        (
            (torch.tensor(interior.coords, dtype=torch.float32, device=device) - origin)
            * scale_factor
        )
        .round()
        .to(torch.int32)
        for interior in polygon.interiors
    ]
    return [exterior], holes
