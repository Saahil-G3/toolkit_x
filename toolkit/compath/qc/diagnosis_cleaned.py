import cv2
import geojson
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import segmentation_models_pytorch as smp

from load import h5, save_geojson
from compath.slide.wsi import WSIManager
from compath.dataloading.slicer import Slicer
from image_tools import get_rgb_colors, get_cmap
from geometry.cv2_tools import get_contours, get_shapely_poly
from geometry.shapely_tools import MultiPolygon, loads, geom_to_geojson

from ._model_metadata import get_metadata

class Diagnosis(Slicer):
    def __init__(self):
        self.data_loading_mode = "cpu"

    def initialise_wsi(self, wsi_path, wsi_type, device):
        wsi = WSIManager(wsi_path).get_wsi_object(wsi_type)
        Slicer.__init__(self, wsi=wsi, device=device)
        self.qc_folder = Path(f"results/{self.wsi._wsi_path.stem}/qc")
        self.qc_folder.mkdir(parents=True, exist_ok=True)

        #Set metadata
        metadata = get_metadata()
        for key in list(metadata.keys()):
            results_path = {
                "geojson": Path(f"{self.qc_folder}/{key}.geojson"),
                "h5": Path(f"{self.qc_folder}/{key}.h5"),
                }
            metadata[key]["results_path"] = results_path
        self.metadata = metadata

        #Check for tissue mask
        if Path(f"{self.qc_folder}/tissue.h5").exists():
            tissue_wkt = h5.load_wkt_dict(Path(f"{self.qc_folder}/tissue.h5"))[
                "combined"
            ]
            tissue_geom = loads(tissue_wkt)
            super().set_tissue_geom(tissue_geom)














































