"""
for documentaion refer to - 
https://docs.pathomation.com/sdk/pma.python.documentation/pma_python.html
"""

import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from pma_python import core
from typing import Union, List, Optional

from toolkit.geometry.shapely_tools import Polygon, MultiPolygon
from toolkit.vision.colors import get_hex_colors, percentage_to_hex_alpha

from toolkit.system.logging_tools import Logger

text_logger = Logger(
    name="pathomation",
    log_to_console=False,
    log_to_txt=True,
    log_to_csv=False,
    add_timestamp=True,
).get_logger()

console_logger = Logger(name="pathomation").get_logger()

from .base_wsi import BaseWSI


class PathomationWSI(BaseWSI):
    def __init__(
        self,
        wsi_path: Path,
        sessionID: str = None,
        tissue_geom: Union[Polygon, MultiPolygon] = None,
    ):
        super().__init__(wsi_path=wsi_path, tissue_geom=tissue_geom)

        self.sessionID = sessionID
        self.wsi_type = "Pathomation"
        self._slideRef = str(self._wsi_path)
        self.dims = core.get_pixel_dimensions(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self._mpp_x, self._mpp_y = core.get_pixels_per_micrometer(
            self._slideRef, zoomlevel=None, sessionID=self.sessionID
        )
        self.mpp = self._mpp_x

        if self._mpp_x != self._mpp_y:
            warnings.warn("mpp_x is not equal to mpp_y.", UserWarning)

        self.zoomlevels = core.get_zoomlevels_list(
            self._slideRef, sessionID=self.sessionID, min_number_of_tiles=0
        )
        self.level_count = len(self.zoomlevels)
        self._set_level_mpp_dict()

        text_logger.info(f"Initiated session with sessionID: {self.sessionID} for WSI at {self._wsi_path}.")

    def get_thumbnail_at_mpp(self, target_mpp=50):
        factor = self.factor_mpp(target_mpp)
        dims = (int(self.dims[0] // factor), int(self.dims[1] // factor))
        return self.get_thumbnail_at_dims(dims)

    def get_thumbnail_at_dims(self, dims):
        thumbnail = core.get_thumbnail_image(
            self._slideRef,
            width=dims[0],
            height=dims[1],
            sessionID=self.sessionID,
            verify=True,
        )
        return thumbnail

    def get_level_for_downsample(self, factor):
        for key, value in self.level_mpp_dict.items():
            if value["factor"] < factor:
                break
        return key

    def get_region_for_slicer(self, coordinate, slice_params):
        x, y = coordinate
        w, h = slice_params["extraction_dims"]
        factor = slice_params["factor1"]
        w_scaled = self.round_to_nearest_even(w * factor)
        h_scaled = self.round_to_nearest_even(h * factor)
        region = self._get_region(x, y, w_scaled, h_scaled, scale=1 / factor).convert(
            "RGB"
        )
        return region

    def _get_region(self, x: int, y: int, w: int, h: int, scale: float = 1):
        region = core.get_region(
            self._slideRef,
            x=x,
            y=y,
            width=w,
            height=h,
            scale=scale,
            sessionID=self.sessionID,
        )
        return region

    def _set_level_mpp_dict(self):
        level_mpp_dict = {}
        for level in self.zoomlevels:
            temp_dict = {}
            mpp_x, mpp_y = core.get_pixels_per_micrometer(
                self._slideRef, zoomlevel=level, sessionID=self.sessionID
            )

            temp_dict["level"] = level
            temp_dict["mpp"] = mpp_x
            # factor to go from original mpp to mpp_x
            temp_dict["factor"] = self.factor_mpp(temp_dict["mpp"])
            temp_dict["dims"] = (
                int(self.dims[0] // temp_dict["factor"]),
                int(self.dims[1] // temp_dict["factor"]),
            )
            level_mpp_dict[self.level_count - level - 1] = temp_dict
            if mpp_x != mpp_y:
                warnings.warn(
                    f"mpp_x is not equal to mpp_y at level {level}", UserWarning
                )

        self.level_mpp_dict = level_mpp_dict

    def _get_slice_wsi_coordinates(self, slice_params):
        coordinates = []
        factor1 = slice_params["factor1"]
        x_lim, y_lim = self.dims
        extraction_dims = slice_params["extraction_dims"]
        stride_dims = slice_params["stride_dims"]
        context_dims = slice_params["context_dims"]

        scaled_extraction_dims = int(extraction_dims[0] * factor1), int(extraction_dims[1] * factor1)
        scaled_stride_dims = int(stride_dims[0] * factor1), int(stride_dims[1] * factor1)
        scaled_context_dims = int(context_dims[0] * factor1), int(context_dims[1] * factor1)

        max_x = x_lim + scaled_context_dims[0]
        max_y = y_lim + scaled_context_dims[1]

        max_x_adj = max_x - scaled_extraction_dims[0]
        max_y_adj = max_y - scaled_extraction_dims[1]

        for x in range(-scaled_context_dims[0], max_x, scaled_stride_dims[0]):
            #x_clipped = min(x, max_x_adj)
            x_clipped = max(0, min(x, max_x_adj))

            for y in range(-scaled_context_dims[1], max_y, scaled_stride_dims[1]):
                #y_clipped = min(y, max_y_adj)
                y_clipped = max(0, min(y, max_y_adj)) 

                coordinates.append(((x_clipped, y_clipped), False))
        return coordinates

    # Pathomation Specific Methods
    def add_annotation(
        self, wkt, layerID=666, classification="Unclassified", fill_opacity=65
    ):
        """
        Adds a single annotation to the slide using the PMA.core module. The annotation is styled with a predefined color and fill opacity, and uploaded to a specified layer.

        Args:
            wkt (str): The geometry of the annotation in Well-Known Text (WKT) format.
            layerID (int, optional): The identifier for the layer to which the annotation will be added. Defaults to 666.
            classification (str, optional): The classification label for the annotation. Defaults to "Unclassified".
            fill_opacity (int, optional): The opacity of the fill color, where 0 is fully transparent and 100 is fully opaque. Defaults to 80.

        Example:
            instance.add_annotation(wkt=gc_mpp2_geom.wkt, classification="Tumor", fill_opacity=90)
        """
        ann = core.dummy_annotation()

        ann["geometry"] = wkt
        ann["lineThickness"] = 3
        ann["color"] = f"#000000FF"
        ann["fillColor"] = (
            f"#B2FF9E{percentage_to_hex_alpha(fill_opacity)}"  # Pale Lime
        )

        add_annotation_output = core.add_annotations(
            slideRef=self._slideRef,
            classification=classification,
            notes=classification,
            anns=ann,
            layerID=layerID,
            sessionID=self.sessionID,
        )

        console_logger.info(f"Add annotation ({self.name}): {add_annotation_output['Code']}")
        text_logger.info(f"Added annotation for: {self.name}.")
        text_logger.info(f"{add_annotation_output}")

    def add_annotations(self, anns, layerID=666, show_progress=True, fill_opacity=65):
        """
        Adds annotations to the slide using the PMA.core module. Each annotation is styled with unique colors and uploaded to a specified layer.

        Args:
            anns (list): A list of dictionaries where each dictionary represents an annotation.
                Each dictionary must contain:
                - `wkt` (str): The geometry of the annotation in Well-Known Text (WKT) format.
                - `classification` (str): The classification label for the annotation.
                Optionally, it can contain:
                - `notes` (str): Additional notes for the annotation. If not provided, it will default to the value of `classification`.
            layerID (int, optional): The identifier for the layer to which the annotations will be added. Defaults to 666.
            show_progress (bool, optional): Whether to display a progress bar during the upload process. Defaults to True.

        Raises:
            ValueError: If any annotation in `anns` is missing the required keys: `wkt` or `classification`.

        Example:
            anns = [
                {"wkt": gc_mpp2_geom.wkt, "classification": gc_mpp2._model_name},
                {"wkt": gc_mpp1_geom.wkt, "classification": gc_mpp1._model_name, "notes": "Important region"},
            ]
            instance.add_annotations(anns, layerID=42, show_progress=False)
        """

        required_keys = {"wkt", "classification"}
        for ann in anns:
            if not required_keys.issubset(ann.keys()):
                raise ValueError(
                    f"Each annotation must contain the keys {required_keys}. Missing in: {ann}"
                )
            if "notes" not in ann:
                ann["notes"] = ann["classification"]

        colors = get_hex_colors(len(anns))
        processed_anns = []

        if show_progress:
            iterator = enumerate(tqdm(anns, desc="Uploading annotations to pma"))
        else:
            iterator = enumerate(anns)

        for idx, ann in iterator:

            fill_color = colors[idx]
            dummy_ann = core.dummy_annotation()
            dummy_ann["lineThickness"] = 3
            dummy_ann["color"] = f"#000000FF"
            dummy_ann["fillColor"] = (
                f"{fill_color}{percentage_to_hex_alpha(fill_opacity)}"
            )

            dummy_ann["geometry"] = ann["wkt"]

            add_annotation_output = core.add_annotations(
                slideRef=self._slideRef,
                classification=ann["classification"],
                notes=ann["notes"],
                anns=dummy_ann,
                layerID=layerID,
                sessionID=self.sessionID,
            )

            console_logger.info(f"Add annotation ({self.name}): {add_annotation_output['Code']}")
            text_logger.info(f"Added annotation for: {self.name}.")
            text_logger.info(f"{add_annotation_output}")

    def clear_annotations(self, layerID=666):
        """
        Clears annotations from a specified layer on the slide using the PMA.core module.

        Args:
            layerID (int, optional): The identifier for the layer from which the annotations will be cleared. Defaults to 666.

        Example:
            instance.clear_annotations(layerID=42)
        """

        clear_annotations_output = core.clear_annotations(
            slideRef=self._slideRef, layerID=layerID, sessionID=self.sessionID
        )
        if clear_annotations_output:
            text_logger.info(f"Cleared annotation for: {self.name}.")
            console_logger.info(f"Cleared annotation at layer {layerID}.")
        else:
            console_logger.warning(f"Unable to clear annotation at layer {layerID}.")

    @staticmethod
    def get_pathomation_sessionID(pmacoreURL, pmacoreUsername, pmacorePassword):
        sessionID = core.connect(
            pmacoreURL=pmacoreURL,
            pmacoreUsername=pmacoreUsername,
            pmacorePassword=pmacorePassword,
            verify=True,
        )
        return sessionID

    @staticmethod
    def get_tray(wsi_paths):
        tray = []
        for wsi_path in wsi_paths:
            tray.append(
                {"Slide Info::Server;Slide Info::File name": "PMA.core;" + wsi_path}
            )
        tray = pd.DataFrame(tray)
        return tray
