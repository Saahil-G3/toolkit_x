import torch
import numpy as np
import geojson
from shapely.wkt import loads
from shapely.geometry import mapping
from shapely.geometry import box as Box
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
# from shapely.validation import make_valid
# from shapely.strtree import STRtree


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


def wkt_to_geojson(wkt_string):
    """
    Converts a WKT string to a GeoJSON feature.

    Args:
        wkt_string (str): A string containing a WKT representation of a geometry.

    Returns:
        geojson.Feature: A GeoJSON feature representation of the geometry.
    """
    poly = loads(wkt_string)
    poly = make_valid(poly)
    geojson_feature = geojson.Feature(geometry=mapping(poly))
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
