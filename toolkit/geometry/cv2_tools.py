import cv2
import geojson
import numpy as np
from cv2 import resize

from toolkit.geometry.shapely_tools import Polygon

"""
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
hierarchy = [[[ 1, -1, -1, -1],  # Contour 0: [Next, Prev, First_Child, Parent]
              [-1, -1, -1,  0],  # Contour 1: [Next, Prev, First_Child, Parent]
              [-1, -1, -1,  0]]] # Contour 2: [Next, Prev, First_Child, Parent]
"""


def get_shapely_poly(
    contours, hierarchy, scale_factor=1, shift_x=0, shift_y=0, process_hierarchy=True
):
    """
    Convert contours into Shapely polygons, optionally processing the hierarchy to handle nested structures.

    Args:
        contours: List of contours, each represented as an array of points.
        hierarchy: Contour hierarchy array containing relationships between contours.
        scale_factor: Scaling factor to adjust the contour coordinates.
        shift_x: Horizontal shift to apply to contour coordinates.
        shift_y: Vertical shift to apply to contour coordinates.
        process_hierarchy: Whether to process contour relationships based on hierarchy.

    Returns:
        List of Shapely `Polygon` objects.
    """

    assert len(contours) > 0, "No contours to process"
    polys = []

    if process_hierarchy:
        idx_map = get_idx_map(contours, hierarchy)

        for outer_idx, inner_idxs in idx_map.items():
            outer_contour = contours[outer_idx]
            if outer_contour.shape[0] < 4:
                continue
            outer_contour = [
                (
                    (point[0][0] * scale_factor) + shift_x,
                    (point[0][1] * scale_factor + shift_y),
                )
                for point in outer_contour
            ]

            holes = []
            if len(inner_idxs) > 0:
                for inner_idx in inner_idxs:
                    inner_contour = contours[inner_idx]
                    if inner_contour.shape[0] < 4:
                        continue
                    inner_contour = [
                        (
                            (point[0][0] * scale_factor) + shift_x,
                            (point[0][1] * scale_factor + shift_y),
                        )
                        for point in inner_contour
                    ]
                    holes.append(inner_contour)

            poly = Polygon(shell=outer_contour, holes=holes)
            polys.append(poly)
    else:
        for contour in contours:
            if contour.shape[0] < 4:
                continue
            contour = [
                (
                    (point[0][0] * scale_factor) + shift_x,
                    (point[0][1] * scale_factor + shift_y),
                )
                for point in contour
            ]
            poly = Polygon(contour)
            polys.append(poly)

    return polys


def get_multipolygon_geojson_feature(
    contours, idx_map, label, color, scale_factor, show_pbar=True
):
    """
    Generate a GeoJSON Feature containing a MultiPolygon from contours and their hierarchical relationships.

    Args:
        contours: List of contours represented as arrays of points.
        idx_map: Mapping of parent contour indices to their child indices.
        label: Label for the GeoJSON feature.
        color: Color property for the feature.
        scale_factor: Scaling factor to adjust contour coordinates.
        show_pbar: Whether to display a progress bar during processing.

    Returns:
        GeoJSON Feature with a MultiPolygon geometry and associated properties.
    """
    geojson_polygons = []

    if show_pbar:
        pbar = tqdm(total=len(idx_map))

    for outer_idx, inner_idxs in idx_map.items():
        outer_contour = contours[outer_idx]
        if outer_contour.shape[0] < 4:
            if show_pbar:
                pbar.update(1)
            continue
        geojson_contour = []
        outer_geojson_contour = get_geojson_contour(
            outer_contour, scale_factor=scale_factor
        )
        geojson_contour.append(outer_geojson_contour)

        if len(inner_idxs) > 0:
            for inner_idx in inner_idxs:
                inner_contour = contours[inner_idx]
                if inner_contour.shape[0] < 4:
                    continue
                inner_geojson_contour = get_geojson_contour(
                    inner_contour, scale_factor=scale_factor
                )
                geojson_contour.append(inner_geojson_contour)

        geojson_polygons.append(geojson_contour)

        if show_pbar:
            pbar.update(1)

    properties = {"objectType": "annotation", "name": label, "color": color}
    geojson_feature = geojson.Feature(
        geometry=geojson.MultiPolygon(geojson_polygons), properties=properties
    )

    return geojson_feature


def get_geojson_contour(contour, scale_factor=1, shift_x=0, shift_y=0):
    """
    Convert a single contour into a GeoJSON-compatible list of coordinates.

    Args:
        contour: A contour represented as an array of points.
        scale_factor: Scaling factor to adjust contour coordinates.
        shift_x: Horizontal shift to apply to contour coordinates.
        shift_y: Vertical shift to apply to contour coordinates.

    Returns:
        List of coordinates in GeoJSON format.
    """

    contour = contour.squeeze(1)
    X = (contour[:, 0] * scale_factor) + shift_x
    Y = (contour[:, 1] * scale_factor) + shift_y

    geojson_contour = []
    for x, y in zip(X, Y):
        geojson_contour.append([x, y])

    geojson_contour = (np.array(geojson_contour).astype(int)).tolist()
    geojson_contour.append(geojson_contour[0])

    return geojson_contour


def get_contours(mask):
    """
    Extract contours and their hierarchy from a binary mask using OpenCV.

    Args:
        mask: Binary image where contours are to be detected.

    Returns:
        contours: List of contours represented as arrays of points.
        hierarchy: Array describing the relationships between contours.
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy


def get_circularity(contour):
    """
    Calculate the circularity of a contour based on its area and perimeter.

    Args:
        contour: A contour represented as an array of points.

    Returns:
        circularity: Circularity of the contour (float).
        area: Area of the contour (float).
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    circularity = (4 * np.pi * area) / (perimeter * perimeter)

    return round(circularity, 3), round(area, 3)


def _get_contour_status(contours, hierarchy):
    """
    Classify contours based on their relationships in the hierarchy.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Dictionary with keys:
            - 'solo': Contours with no parent or child.
            - 'only_parent': Contours with no parent but having children.
            - 'only_daughter': Contours with a parent but no children.
            - 'parent_daughter': Contours with both a parent and children.
    """
    solo = []
    only_parent = []
    only_daughter = []
    parent_daughter = []

    # Traverse each contour and classify
    for idx, contour in enumerate(contours):
        h = hierarchy[0][idx]

        parent_idx = h[3]  # Index of the parent contour
        child_idx = h[2]  # Index of the first child contour

        if parent_idx == -1 and child_idx == -1:
            solo.append(idx)
        elif parent_idx == -1 and child_idx != -1:
            only_parent.append(idx)
        elif parent_idx != -1 and child_idx == -1:
            only_daughter.append(idx)
        elif parent_idx != -1 and child_idx != -1:
            parent_daughter.append(idx)

    assert len(solo) + len(only_parent) + len(only_daughter) + len(
        parent_daughter
    ) == len(contours)

    contour_status = {}
    contour_status["solo"] = solo
    contour_status["only_parent"] = only_parent
    contour_status["only_daughter"] = only_daughter
    contour_status["parent_daughter"] = parent_daughter

    return contour_status


def get_idx_map(contours, hierarchy):
    """
    Create a mapping of contour indices based on their hierarchical relationships.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Dictionary mapping parent contour indices to lists of child contour indices.
    """
    contour_status = _get_contour_status(contours, hierarchy)
    idx_map = {}
    idx_map = dict.fromkeys(contour_status["solo"], [])
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["only_daughter"], hierarchy
    )
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["parent_daughter"], hierarchy
    )

    return idx_map


def _get_hierarchy_idx_map(idx_map, contour_status, hierarchy):
    """
    Update the contour index map by processing a specific set of contour statuses.

    Args:
        idx_map: Existing mapping of parent indices to child indices.
        contour_status: Indices of contours to be processed.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Updated mapping of parent indices to child indices.
    """
    for idx in contour_status:
        h = hierarchy[0][idx]

        parent_idx = h[3]  # Index of the parent contour
        child_idx = h[2]  # Index of the first child contour

        found = False
        while not found:
            parent = hierarchy[0][parent_idx]
            if parent[-1] == -1:
                found = True
            else:
                parent_idx = parent[-1]

        if parent_idx in idx_map:
            idx_map[parent_idx].append(idx)
        else:
            idx_map[parent_idx] = []
            idx_map[parent_idx].append(idx)

    return idx_map


"""
def _get_contour_status(contours, hierarchy):
    solo = []
    only_parent = []
    only_daughter = []
    parent_daughter = []

    # Traverse each contour and classify
    for idx, contour in enumerate(contours):
        h = hierarchy[0][idx]
    
        parent_idx = h[3]   # Index of the parent contour
        child_idx = h[2]    # Index of the first child contour

        if parent_idx == -1 and child_idx == -1:
            solo.append(idx)
        elif parent_idx == -1 and child_idx != -1:
            only_parent.append(idx)
        elif parent_idx != -1 and child_idx == -1:
            only_daughter.append(idx)
        elif parent_idx != -1 and child_idx != -1:
            parent_daughter.append(idx)

    assert len(solo)+len(only_parent)+len(only_daughter)+len(parent_daughter) == len(contours)
    
    contour_status   = {}
    contour_status['solo'] = solo
    contour_status['only_parent'] = only_parent 
    contour_status['only_daughter'] = only_daughter
    contour_status['parent_daughter'] = parent_daughter
    
    return contour_status
"""


def _get_wkt_str(X, Y):
    """
    Generate a Well-Known Text (WKT) representation of a polygon from coordinate arrays.

    Args:
        X: Array of x-coordinates.
        Y: Array of y-coordinates.

    Returns:
        WKT string representing the polygon.
    """
    wkt = str()
    for x, y in zip(X, Y):
        wkt = wkt + f"{int(x)} {int(y)},"
    wkt = wkt + f"{int(X[0])} {int(Y[0])},"
    wkt = f"({wkt[:-1]})"
    return wkt


def _get_master_wkt(wkt_list):
    """
    Combine multiple WKT strings into a master WKT string for a polygon.

    Args:
        wkt_list: List of WKT strings.

    Returns:
        Master WKT string representing a polygon with multiple rings.
    """
    master_wkt = str()
    for wkt in wkt_list:
        master_wkt = f"{master_wkt}{wkt},"

    master_wkt = f"POLYGON ({master_wkt[:-1]})"
    return master_wkt


def process_contour_hierarchy(
    contours,
    hierarchy,
    contour_mpp,
    origin_shift=(0, 0),
    rescale_factor=1,
    process_daughters=True,
):
    """
    Process contour hierarchy to compute master WKT strings, areas, and circularities for contours and their children.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.
        contour_mpp: Microns per pixel (mpp) at which contours were calculated.
        origin_shift: Shift to apply to the origin of coordinates (x, y).
        rescale_factor: Scaling factor to adjust coordinates.
        process_daughters: Whether to process child contours.

    Returns:
        List of dictionaries with keys:
            - 'master_wkt': WKT string for the polygon.
            - 'area': Area of the contour in microns².
            - 'circularity': Circularity of the contour.
    """
    contour_status = _get_contour_status(contours, hierarchy)
    idx_map = {}
    idx_map = dict.fromkeys(contour_status["solo"], [])
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["only_daughter"], hierarchy
    )
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["parent_daughter"], hierarchy
    )

    total_contours = 0
    for key, value in idx_map.items():
        total_contours += 1
        total_contours += len(value)

    assert total_contours == len(
        contours
    ), "Total Contours processed not equal to number of input contours"

    master_wkt_list = []
    for contour_idx, value in idx_map.items():
        wkt_list = []
        contour = contours[contour_idx]
        if cv2.contourArea(contour) < 2:
            continue

        circularity, area = get_circularity(contour)
        master_contour_area = (area) * (contour_mpp**2)

        X = contour[:, :, 0] * rescale_factor + origin_shift[0]
        Y = contour[:, :, 1] * rescale_factor + origin_shift[1]

        wkt_list.append(_get_wkt_str(X, Y))
        if process_daughters:
            if len(value) > 0:
                for contour_idx in value:
                    contour = contours[contour_idx]
                    X = contour[:, :, 0] * rescale_factor + origin_shift[0]
                    Y = contour[:, :, 1] * rescale_factor + origin_shift[1]
                    wkt_list.append(_get_wkt_str(X, Y))

        master_wkt = _get_master_wkt(wkt_list)
        master_wkt_list.append(
            {
                "master_wkt": master_wkt,
                "area": master_contour_area,
                "circularity": circularity,
            }
        )

    return master_wkt_list


def get_parent_daughter_idx_map(contours, hierarchy):
    """
    Generate a mapping of parent contour indices to their child indices based on the hierarchy.

    Args:
        contours: List of contours represented as arrays of points.
        hierarchy: Contour hierarchy array containing relationships.

    Returns:
        Dictionary mapping parent contour indices to lists of child contour indices.
    """

    contour_status = _get_contour_status(contours, hierarchy)
    idx_map = {}
    idx_map = dict.fromkeys(contour_status["solo"], [])
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["only_daughter"], hierarchy
    )
    idx_map = _get_hierarchy_idx_map(
        idx_map, contour_status["parent_daughter"], hierarchy
    )

    return idx_map
