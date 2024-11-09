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
    GeometryCollection    
)

is_collection = {}
is_collection['Point'] = False
is_collection['Polygon'] = False
is_collection['LineString'] = False
is_collection['LinearRing'] = False
is_collection['MultiPoint'] = True
is_collection['MultiPolygon'] = True
is_collection['MultiLineString'] = True
is_collection['GeometryCollection'] = True

from shapely.prepared import prep as prep_geom_for_query

#import cv2
#import math
#import random
#import numpy as np
#from tqdm.auto import tqdm
#from shapely.geometry import shape as Shape
#from shapely.ops import unary_union
#from shapely import wkt
#from shapely.validation import make_valid
#from shapely.strtree import STRtree


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







