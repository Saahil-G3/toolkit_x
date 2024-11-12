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
    geojson_feature = geojson.Feature(geometry=mapping(geom))
    return geojson_feature


def wkt_to_geojson(wkt_string):
    poly = loads(wkt_string)
    poly = make_valid(poly)
    geojson_feature = geojson.Feature(geometry=mapping(poly))
    return geojson_feature


def get_box(x, y, height, width):
    return Box(x, y, x + height, y + width)


def flatten_geom_collection(geom):
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
    if origin is None:
        origin = torch.tensor([0, 0], dtype=torch.float32, device=device)
    exterior = (torch.tensor(polygon.exterior.coords, dtype=torch.float32, device=device) - origin) * scale_factor
    exterior = exterior.round().to(torch.int32)
    
    holes = [
        ((torch.tensor(interior.coords, dtype=torch.float32, device=device) - origin) * scale_factor).round().to(torch.int32)
        for interior in polygon.interiors
    ]
    return [exterior], holes