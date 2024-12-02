import numpy as np
from toolkit.vision.plotting import plot_overlay
from toolkit.geometry.shapely_tools import get_numpy_mask_from_geom

class WSIVisualiser():
    def __init__(self):
        self.wsi = None

    def set_wsi_for_vis(self, wsi):
        self.wsi = wsi

    def overlay_geom_on_wsi(self, geom, target_mpp=50, **kwargs):
        if target_mpp < 5:
            raise ValueError(
                f"Attempting to plot WSI at {target_mpp}, use target_mpp>5"
            )

        factor = self.wsi.factor_mpp(target_mpp)
        thumbnail = self.wsi.get_thumbnail_at_mpp(target_mpp)
        thumbnail = np.array(thumbnail.convert("RGB"))
        mask_dims = thumbnail.shape[:2]
        mask = get_numpy_mask_from_geom(
            geom=geom, mask_dims=mask_dims, origin=(0,0), scale_factor=1 / factor
        )
        plot_overlay(thumbnail, mask, **kwargs)