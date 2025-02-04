from pma_python import core

from .tiffslide import TiffSlideWSI
from .pathomation import PathomationWSI

from toolkit.pathomics.caib.wsi import PathomationCAIBWSI

class WSIManager:
    supported_wsi_types = {"TiffSlide", "Pathomation", "PathomationCAIBWSI"}
    wsi_classes = {
        "TiffSlide": TiffSlideWSI,
        "Pathomation": PathomationWSI,
        "PathomationCAIBWSI": PathomationCAIBWSI,
    }

    def __init__(
        self, wsi_path, wsi_type="TiffSlide", tissue_geom=None, sessionID=None, s3=None
    ):
        """
        Initializes the WSIManager with the provided WSI path, type, and optional metadata.

        Args:
            wsi_path (str): Path to the whole slide image.
            wsi_type (str): Type of WSI to manage ('TiffSlide' or 'Pathomation').
            session_id (str, optional): Session ID for Pathomation WSIs.
            tissue_geom (object, optional): Tissue geometry associated with the WSI.
        """
        if wsi_type not in self.supported_wsi_types:
            raise ValueError(
                f"Unsupported wsi_type '{wsi_type}'. Supported types are: {self.supported_wsi_types}"
            )

        self.wsi_type = wsi_type
        self.wsi_path = wsi_path
        self.sessionID = sessionID
        self.tissue_geom = tissue_geom
        self.s3 = s3
        self.wsi = self._get_wsi()

    def _get_wsi(self):
        """
        Returns an object representing the WSI based on the specified type.

        Returns:
            object: An instance of TiffSlideWSI, PathomationWSI or "PathomationCAIBWSI".
        """
        wsi_class = self.wsi_classes[self.wsi_type]
        
        kwargs = {}
        
        if self.wsi_type == "Pathomation" or self.wsi_type == "PathomationCAIBWSI":
            kwargs["sessionID"] = self.sessionID
            
        if self.wsi_type == "PathomationCAIBWSI":
            kwargs["s3"] = self.s3
            kwargs["wsi_name"] = self.wsi_path
            kwargs["tissue_geom"] = self.tissue_geom

        else:
            kwargs["wsi_path"] = self.wsi_path
            kwargs["tissue_geom"] = self.tissue_geom

        return wsi_class(**kwargs)
