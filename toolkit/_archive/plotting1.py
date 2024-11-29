def contour_to_array(contour, patch_height, patch_width, fill_number=1):
    """
    Convert a contour to a binary array.

    Args:
    - contour: Array of contour points (N, 2).
    - patch_height: Height of the output binary array.
    - patch_width: Width of the output binary array.
    - fill_number: Value to fill inside the contour. Default is 1.

    Returns:
    - seg_mask: Binary mask with the contour filled.
    """

    from skimage.draw import polygon

    seg_mask = np.zeros((patch_height, patch_width))
    rr, cc = polygon(contour[:, 1], contour[:, 0])
    seg_mask[rr, cc] = fill_number

    return seg_mask

def get_bounding_box_cords(single_channel_mask):
    """
    Compute the bounding box coordinates for a single-channel mask.

    Args:
    - single_channel_mask: Binary mask where the object is represented as non-zero pixels.

    Returns:
    - bounding_box: List containing the [min_row, min_col, max_row, max_col] coordinates of the bounding box.
    """

    nonzero_indices = np.nonzero(single_channel_mask)
    min_row, min_col = np.min(nonzero_indices[0]), np.min(nonzero_indices[1])
    max_row, max_col = np.max(nonzero_indices[0]), np.max(nonzero_indices[1])

    return [min_row, min_col, max_row, max_col]

def convert_geojson_contour(X, Y):
    """
    Convert two lists of X and Y coordinates into a GeoJSON-style contour array.
    Args:
    - X (list): List of X-coordinates.
    - Y (list): List of Y-coordinates.
    Returns:
    - numpy.ndarray: Contour coordinates as a NumPy array.
    """
    cnt_list = []
    for x, y in zip(X, Y):
        cnt_list.append([x, y])

    return np.array(cnt_list).astype(int)


def get_wkt(X, Y):
    """
    Convert two lists of X and Y coordinates into a Well-Known Text (WKT) polygon string.
    Args:
    - X (list): List of X-coordinates.
    - Y (list): List of Y-coordinates.
    Returns:
    - str: Polygon in WKT format.
    """
    wkt = str()

    for x, y in zip(X, Y):
        wkt = wkt + f"{int(x)} {int(y)},"

    wkt = wkt + f"{int(X[0])} {int(Y[0])},"
    wkt = wkt[:-1]
    wkt = f"POLYGON (( {wkt} ))"

    return wkt