from typing import Any
from .dint import DOOCSInterface, dump_fdl


class ImageAnalysisServerFacade:
    def __init__(self, screen_name: str, di: DOOCSInterface | None = None) -> None:
        self.screen_name = screen_name
        self.di = di or DOOCSInterface()
        self._fdl_stem = f"XFEL.DIAG/IMAGEANALYSIS/{screen_name}/{{}}"

    def get_xroi_clipping(self) -> tuple[int, int]:
        xmin = self.di.get_value(self._fdl_stem.format("ROI.CLIPPING.XMIN"))
        xmax = self.di.get_value(self._fdl_stem.format("ROI.CLIPPING.XMAX"))
        return xmin, xmax

    def get_yroi_clipping(self) -> tuple[int, int]:
        ymin = self.di.get_value(self._fdl_stem.format("ROI.CLIPPING.YMIN"))
        ymax = self.di.get_value(self._fdl_stem.format("ROI.CLIPPING.YMAX"))
        return ymin, ymax

    def activate_gain_control(self) -> None:
        self.di.set_value(self._fdl_stem.format("ANALYSIS.MODE_GAIN_CONTROL"), 1)

    def set_clipping(self, *, on: bool) -> None:
        self.di.set_value(self._fdl_stem.format("ROI.CLIPPING.ENABLED"), int(on))

    def apply_roi(self) -> None:
        # What does this do that clipping doesn't?
        self.di.set_value(self._fdl_stem.format("ROI.ENABLED"), 1)

    def is_active(self) -> bool:
        """If the image analysis server is doing something, e.g. applying automatic gain control."""
        return self.di.get_value(self._fdl_stem.format("ANALYSIS.MODE_INACTIVE")) != 1
    
    def dump(self):
        return dump_fdl(self._fdl_stem.format("*"))

