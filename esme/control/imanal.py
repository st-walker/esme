from typing import Any
from .dint import DOOCSInterface

INTERESTING_PROPERTIES = ["ROI.CLIPPING.XMIN",
                          "ROI.CLIPPING."]

class ImageAnalysisServerFacade:
    def __init__(self, screen_name: str, di: DOOCSInterface | None = None) -> None:
        self.screen_name = screen_name
        self.di = di or DOOCSInterface()
        self.fdl_stem = f"XFEL.DIAG/IMAGEANALYSIS/{screen_name}/{{}}"

    def get_xroi_clipping(self) -> tuple[int, int]:
        xmin = self.di.get_value(self.fdl_stem.format("ROI.CLIPPING.XMIN"))
        xmax = self.di.get_value(self.fdl_stem.format("ROI.CLIPPING.XMAX"))
        return xmin, xmax

    def get_yroi_clipping(self) -> tuple[int, int]:
        ymin = self.di.get_value(self.fdl_stem.format("ROI.CLIPPING.YMIN"))
        ymax = self.di.get_value(self.fdl_stem.format("ROI.CLIPPING.YMAX"))
        return ymin, ymax

    def activate_gain_control(self) -> None:
        self.di.set_value(self.fdl_stem.format("ANALYSIS.MODE_GAIN_CONTROL"), 1)

    def set_clipping(self, *, on: bool) -> None:
        self.di.set_value(self.fdl_stem.format("ROI.CLIPPING.ENABLED"), int(on))

    def apply_roi(self) -> None:
        # What does this do that clipping doesn't?
        self.di.set_value(self.fdl_stem.format("ROI.ENABLED"), 1)

    def is_active(self) -> bool:
        """If the image analysis server is doing something, e.g. applying automatic gain control."""
        return self.di.get_value(self.fdl_stem.format("ANALYSIS.MODE_INACTIVE")) != 1

    def dump(self, filter_arrays=True) -> dict[str, Any]:        
        addresses = []
        values = []
        stub = f"XFEL.DIAG/IMAGEANALYSIS/{screen_name}/{{}}"
        for name in pydoocs.names(stub.format("*")):
            prop, *_ = name.split()
            address = stub.format(prop)
        try:
            ddata = pydoocs.read(address)
        except pydoocs.DoocsException:
            pass
        is_array_type = data["TYPE"].startswith("A_") or data["TYPE"] == "SPECTUM"
        if is_array_type and filter_arrays:
            addresses.append(address)
            values.append(ddata["data"])
        return dict(zip(addresses, values))