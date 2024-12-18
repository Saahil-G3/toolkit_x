from pathlib import Path
from toolkit.pathomics.wsi.pathomation import PathomationWSI

class PathomationCAIBWSI(PathomationWSI):
    def __init__(self, wsi_name, sessionID=None, s3=None):

        self.wsi_name = Path(wsi_name)
        self.wsi_path = self.get_wsi_path_from_name(self.wsi_name)
        PathomationWSI.__init__(self, wsi_path=self.wsi_path, sessionID=sessionID)
        self.s3 = s3
        
    def download_wsi(self, bucket_name="caib-wsi"):
        self.s3.find_key(str(self.wsi_name), "caib-wsi")
        self.s3.download_file(bucket_name, self.s3.queried_keys[str(self.wsi_name)][0])

    @staticmethod
    def get_wsi_path_from_name(wsi_name: str):
        wsi_name = str(wsi_name)
        part1 = wsi_name[:16]
        part2 = wsi_name[:21]
        part3 = Path(wsi_name).stem
        slideRef = Path("CAIB_WSI") / "CAIB" / part1 / part2 / part3 / wsi_name
        return str(slideRef)




