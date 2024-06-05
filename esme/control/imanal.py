import numpy as np
from dint import DOOCSInterface


class ImageAnalysisServerFacade:
    def __init__(self, screen_name: str, di: DOOCSInterface | None = None) -> None:
        self.screen_name = screen_name
        self.di = di or DOOCSInterface()
        self.fdl_stem = f"XFEL.DIAG/IMAGEANALYSIS/{screen_name}/{{}}"

    def get_roi_slice(self) -> tuple[slice, slice]:
        xmin, xmax = self.get_xroi_clipping(self.screen_name)
        ymin, ymax = self.get_yroi_clipping(self.screen_name)
        return np.s_[ymin:ymax, xmin:xmax]

    def get_xroi_clipping(self) -> tuple[int, int]:
        xmin = self.di.get(self.fdl_stem.format("ROI.CLIPPING.XMIN"))
        xmax = self.di.get(self.fdl_stem.format("ROI.CLIPPING.XMAX"))
        return xmin, xmax

    def get_yroi_clipping(self) -> tuple[int, int]:
        ymin = self.di.get(self.fdl_stem.format("ROI.CLIPPING.YMIN"))
        ymax = self.di.get(self.dfl_stem.format("ROI.CLIPPING.YMAX"))
        return ymin, ymax

    def activate_gain_control(self) -> None:
        self.di.write(self.fdl_stem.format("ANALYSIS.MODE_GAIN_CONTROL"), 1)

    def apply_clipping(self) -> None:
        self.di.write(self.fdl_stem.format("ROI.CLIPPING.ENABLED"), 1)

    def apply_roi(self) -> None:
        # What does this do that clipping doesn't?
        self.di.write(self.fdl_stem.format("ROI.ENABLED"), 1)

    def is_active(self) -> bool:
        """If the image analysis server is doing something, e.g. applying automatic gain control."""
        return self.di.get(self.fd1_stem.format("ANALYSIS.MODE_INACTIVE")) != 1
