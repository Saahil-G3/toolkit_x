from pma_python import core

from ._tiffslide import TiffSlideWSI
from ._pathomation import PathomationWSI

def get_pathomation_sessionID(pmacoreURL, pmacoreUsername, pmacorePassword):
    sessionID = core.connect(
        pmacoreURL=pmacoreURL,
        pmacoreUsername=pmacoreUsername,
        pmacorePassword=pmacorePassword,
        verify=True,
    )
    return sessionID

class WSIManager:
    supported_wsi_types = {"TiffSlide", "Pathomation"}
    wsi_classes = {
        "TiffSlide": TiffSlideWSI,
        "Pathomation": PathomationWSI,
    }

    def __init__(self, wsi_path, wsi_type="TiffSlide", sessionID=None, tissue_geom=None):
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
        self.wsi = self._get_wsi()

    def _get_wsi(self):
        """
        Returns an object representing the WSI based on the specified type.

        Returns:
            object: An instance of TiffSlideWSI or PathomationWSI.
        """
        wsi_class = self.wsi_classes[self.wsi_type]
        kwargs = {
            "wsi_path": self.wsi_path,
            "tissue_geom": self.tissue_geom,
        }
        if self.wsi_type == "Pathomation":
            kwargs["sessionID"] = self.sessionID
        
        return wsi_class(**kwargs)

