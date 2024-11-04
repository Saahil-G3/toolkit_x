import cv2
import math
import random
import geojson
import numpy as np
from tqdm.auto import tqdm
from shapely.geometry import box

from shapely.geometry import (
    LineString, Polygon, MultiPolygon, MultiLineString, Point, MultiPoint, 
    GeometryCollection, LinearRing
)

from shapely.geometry import shape as Shape
from shapely.geometry import mapping
#from shapely.ops import unary_union
from shapely import wkt
from shapely.wkt import loads
from shapely.validation import make_valid

def wkt_to_geojson(wkt_string): 
    poly = loads(wkt_string)
    poly = make_valid(poly)
    geojson_feature = geojson.Feature(geometry=mapping(poly))
    return geojson_feature

def flatten_geoms(geom):
    """
    Flatten a list of geometries, ensuring that all elements are Polygons.
    """
    flat_geoms = []
    if isinstance(geom, MultiPolygon):
        for geom in geom.geoms:
            flat_geoms.append(geom)
    elif isinstance(geom, Polygon):
        flat_geoms.append(geom)
    elif isinstance(geom, GeometryCollection):
        for geom in geom.geoms:
            flat_geoms.append(geom)
    else:
        print(f"{geom.geom_type}")
        flat_geoms.append(geom)
            
    return flat_geoms

def fix_geom(geom, max_attempts=10):
    attempts = 0
    while not geom.is_valid and attempts < max_attempts:
        #print(f"Attempt {attempts + 1}: Polygon is invalid, trying to fix...")
        
        geom = geom.buffer(0)
        
        if not geom.is_valid:
            geom = make_valid(polygon)
        if attempts==8:
            geom = remove_duplicates_valid(geom)
        attempts += 1
    return geom

def get_geom_coordinates(geom):
    contours = []
    holes = []
    
    if isinstance(geom, Point):
        contours.append([[geom.x, geom.y]])  # Wrap coordinates in a list
        
    elif isinstance(geom, MultiPoint):
        for point in geom.geoms:
            contours.append([point.x, point.y])  # Single list for each point
            
    elif isinstance(geom, LineString):
        contours.append(list(geom.coords))  # Keep coordinates as a list of tuples
        
    elif isinstance(geom, MultiLineString):
        for line in geom.geoms:
            contours.append(list(line.coords))  # Append each line's coordinates
            
    elif isinstance(geom, LinearRing):
        contours.append(list(geom.coords))  # Add LinearRing coordinates as one list

    elif isinstance(geom, Polygon):
        contours.append(list(geom.exterior.coords))  # Exterior coordinates as one list
        
        for interior in geom.interiors:
            holes.append(list(interior.coords))  # Append each hole's coordinates

    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            poly_contours, poly_holes = get_geom_coordinates(poly)
            contours.extend(poly_contours)  # Add polygon contours, one level deep
            holes.extend(poly_holes)        # Add polygon holes, one level deep

    elif isinstance(geom, GeometryCollection):
        for geometry in geom.geoms:
            geom_contours, geom_holes = get_geom_coordinates(geometry)
            contours.extend(geom_contours)  # Add geometry contours, one level deep
            holes.extend(geom_holes)        # Add geometry holes, one level deep
            
    return contours, holes


def sample_from_geom(geom, geom_limit, patch_size, overlap, n_samples):
    coords_list=[]
    start_x,start_y,stop_x,stop_y=get_geom_slicing_bounds(geom, geom_limit, patch_size)
    sampled=0
    while sampled!=n_samples:
        x=random.randint(start_x, stop_x)
        y=random.randint(start_y, stop_y)
    
        sampled_box=get_box(x,y,patch_size,patch_size)
        if geom.intersects(sampled_box):
            coords_list.append((x,y))
            sampled+=1
    return coords_list

def slice_geom(geom, geom_limit, patch_size, overlap):
    coords_list=[]
    start_x,start_y,stop_x,stop_y=get_geom_slicing_bounds(geom, geom_limit, patch_size)
    for x in range(start_x,stop_x, int(overlap)):
        for y in range(start_y,stop_y, int(overlap)):
            sampled_box=get_box(x,y,patch_size,patch_size)
            if geom.intersects(sampled_box):
                coords_list.append((x,y))
    return coords_list

def sample_box_touching_geom(geom,height,width):
    condition=True
    while condition:
        x,y=sample_point_within_geom(geom)
        random_box=get_box(x, y, height, width)

        if geom.intersection(random_box):
            condition=False
    return random_box

def get_geom_slicing_bounds(geom, geom_limit, patch_size):
    min_x, min_y, max_x, max_y = geom.bounds
    lim_min_x, lim_min_y, lim_max_x, lim_max_y = geom_limit.bounds
    
    width=int(max_x - min_x)
    height=int(max_y - min_y)

    max_dim=max(width, height)
    diff=abs(patch_size-max_dim)
    delta=diff+patch_size+random.randint(0,patch_size)

    start_x=min_x-delta
    if start_x<lim_min_x:
        start_x=min_x
    
    start_y=min_y-delta
    if start_y<lim_min_y: 
        start_y=min_y
    
    stop_x=max_x+delta
    if stop_x>lim_max_x:
        stop_x=max_x
        
    stop_y=max_y+delta
    if stop_y>lim_max_y:
        stop_y=max_y
    return int(start_x),int(start_y),int(stop_x),int(stop_y)

def sample_point_within_geom(geom):
    minx, miny, maxx, maxy = geom.bounds
    while True:
        x_rand = int(random.uniform(minx, maxx))
        y_rand = int(random.uniform(miny, maxy))
        random_point = Point([(x_rand, y_rand)])
        if geom.contains(random_point):
            return (x_rand, y_rand)

def get_geoms_from_mask(mask, rescale):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geoms = []
    for contour in contours:
        if len(contour)<4:
            continue
        contour=(contour*rescale).astype(int)
        contour_points = [tuple(point[0]) for point in contour]
        geom = Polygon(contour_points)
        geoms.append(geom)
    return geoms

def get_background(geom):
    bounds=geom.bounds
    bounding_box=box(bounds[0] - 1, bounds[1] - 1, bounds[2] + 1, bounds[3] + 1)
    background=bounding_box.difference(geom)
    return background

def get_box(x, y, width, height):
    return box(x, y, x + width, y + height)

def validate_and_repair(polygon):
    if not polygon.is_valid:
        #print("Invalid polygon detected.")
        polygon = make_valid(polygon)
        if not polygon.is_valid:
            raise ValueError("Polygon could not be repaired.")
    return polygon

def remove_duplicates_valid(polygon):
    coords = list(dict.fromkeys(polygon.exterior.coords))
    return Polygon(coords)

def fit_circle(polygon):
    # Ensure the input is a Shapely Polygon
    if not isinstance(polygon, Polygon):
        raise ValueError("Input must be a Shapely Polygon.")

    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Compute maximum distance from centroid to polygon boundary
    max_distance = 0
    for point in polygon.exterior.coords:
        dist = centroid.distance(Point(point))
        if dist > max_distance:
            max_distance = dist

    # Create a circle as a Shapely Polygon
    circle = Point(centroid.x, centroid.y).buffer(max_distance)

    return circle

def get_circularity(polygon):
    area = polygon.area
    perimeter = polygon.length
    
    if perimeter == 0:
        return 0.0
    
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    
    return circularity

def get_maj2min_ratio(polygon):
    '''
    returns:
        maj_axis, min_axis, maj_axis/min_axis
    '''
    poly_min_rect = polygon.minimum_rotated_rectangle
    
    minx, miny, maxx, maxy = poly_min_rect.bounds
    
    xdiff = maxx - minx
    ydiff = maxy - miny
    
    maj_axis = max(xdiff, ydiff)
    min_axis = min(xdiff, ydiff)
    return maj_axis, min_axis, maj_axis/min_axis

def remove_duplicates(polys):
    '''
    Removes duplicate polygons from a list of polygons.
    '''
    seen_polygons = {}
    unique_polys = []

    # Iterate through the list and keep track of unique polygons
    for polygon in polys:
        wkt_representation = wkt.dumps(polygon)
        if wkt_representation not in seen_polygons:
            seen_polygons[wkt_representation] = polygon
            unique_polys.append(polygon)
    
    return unique_polys

def find_polygon_relationships(polys1, polys2, show_pbar = False):
    '''
    Finds the overlapping relations between two lists of polygon
    '''
    
    polys2_to_remove = set()
    polys1_to_remove = set()

    common_polys = []

    if show_pbar:
        pbar = tqdm(total = len(polys1))
        
    for i, poly1 in enumerate(polys1):
        has_intersection = False
        intersecting_polys = []
    
        for j, poly2 in enumerate(polys2):
        
            inter_area = poly1.convex_hull.intersection(poly2.convex_hull).area
        
            if inter_area>0:
                has_intersection = True
                inter_bool = poly1.convex_hull.intersects(poly2.convex_hull)
                intersecting_polys.append(poly2)
                polys2_to_remove.add(j)
    
        if has_intersection:
            intersecting_polys.append(poly1)
            common_polys.append(intersecting_polys)
            polys1_to_remove.add(i)

        if show_pbar:
            pbar.update()
    unique_to_polys1 = [item for index, item in enumerate(polys1) if index not in polys1_to_remove]
    unique_to_polys2 = [item for index, item in enumerate(polys2) if index not in polys2_to_remove]
    
    return common_polys, unique_to_polys1, unique_to_polys2
    

def get_intersection_map(polys):
    '''
    Gives the index pairs which are overlapping
    '''
    intersecting_idx = []
    for idx1, poly1 in enumerate(polys):
        for idx2, poly2 in enumerate(polys):
            if idx1==idx2:
                continue
        
            inter_area = poly1.intersection(poly2).area
            if inter_area>0:
                intersecting_idx.append([idx1,idx2])
    
    return intersecting_idx
"""
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd

# Define the original polygon
polygon_wkt = 'POLYGON ((82540 147490, 82390 147520, 82360 147530, 82280 147570, 82180 147670, 81940 148120, 81920 148210, 81920 148470, 82650 148480, 82980 148480, 83400 148470, 83410 148420, 83400 148300, 83360 148160, 83310 148010, 83290 147960, 83230 147840, 83170 147760, 83100 147680, 82990 147570, 82940 147540, 82900 147520, 82840 147500, 82760 147490, 82540 147490))'
polygon = Polygon([(82540, 147490), (82390, 147520), (82360, 147530), (82280, 147570), (82180, 147670), 
                   (81940, 148120), (81920, 148210), (81920, 148470), (82650, 148480), (82980, 148480), 
                   (83400, 148470), (83410, 148420), (83400, 148300), (83360, 148160), (83310, 148010), 
                   (83290, 147960), (83230, 147840), (83170, 147760), (83100, 147680), (82990, 147570), 
                   (82940, 147540), (82900, 147520), (82840, 147500), (82760, 147490), (82540, 147490)])

# 1. Get the convex hull of the polygon
convex_hull = polygon.convex_hull

# 2. Approximate an ellipse using a simple circle around the convex hull's bounding box
minx, miny, maxx, maxy = convex_hull.bounds
center = ((minx + maxx) / 2, (miny + maxy) / 2)
radius_x = (maxx - minx) / 2
radius_y = (maxy - miny) / 2

# Generate a circle-like polygon using the ellipse formula
theta = np.linspace(0, 2 * np.pi, 100)
x = center[0] + radius_x * np.cos(theta)
y = center[1] + radius_y * np.sin(theta)

# Create the ellipse polygon
ellipse_coords = list(zip(x, y))
ellipse_polygon = Polygon(ellipse_coords)

# 3. Plot the original polygon and the elliptical polygon
gdf_original = gpd.GeoDataFrame({'geometry': [polygon]})
gdf_ellipse = gpd.GeoDataFrame({'geometry': [ellipse_polygon]})

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot original polygon
gdf_original.plot(ax=ax[0], color='blue', edgecolor='k')
ax[0].set_title('Original Polygon')

# Plot ellipse (approximated circular polygon)
gdf_ellipse.plot(ax=ax[1], color='red', edgecolor='k')
ax[1].set_title('Elliptical Approximation')

plt.show()
"""
