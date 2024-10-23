from typing import Any
from .dint import DOOCSInterface, dump_fdl, DOOCSAddress


class ImageAnalysisServerFacade:
    def __init__(self, screen_name: str, di: DOOCSInterface | None = None) -> None:
        self.screen_name = screen_name
        self.di = di or DOOCSInterface()
        self._fdl_stem = f"XFEL.DIAG/IMAGEANALYSIS/{screen_name}/{{}}"

    def _analysis_address(self, suffix: str) -> str:
        return self._fdl_stem.format(f"ANALYSIS.{suffix}")

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
    
    def set_background_count(self, nbg: int) -> None:
        self.di.set_value(self._analysis_address("BACKGROUND_COUNT_MAX"), nbg)

    def set_sampling_count(self, nim: int) -> None:
        self.di.set_value(self._analysis_address("SAMPLING_COUNT_MAX"), nim)

    def accumulate_background(self) -> None:
        self.di.set_value(self._analysis_address("MODE_BACKGROUND"), 1)

    def set_inactive(self) -> None:
        self.di.set_value(self._analysis_address("MODE_INACTIVE"), 1)

    def start_sampling(self) -> None:
        self.di.set_value(self._analysis_address("MODE_SAMPLING"), 1)

    def start_sampling_with_raw_data(self) -> None:
        self.di.set_value(self._analysis_address("MODE_SAMPLING"), 2)

    def set_subtract_background(self, *, do_subtract: bool) -> None:
        self.di.set_value(self._analysis_address("BACKGROUND_SET_ENABLED"), int(do_subtract))

    def get_gauss_mean_x(self) -> tuple[float, float]:
        return (self.di.get_value(self._analysis_address("X.GAUSS_MEAN")),
            self.di.get_value(self._analysis_address("X.GAUSS_MEAN_ERR")))
        
    def get_gauss_mean_y(self) -> tuple[float, float]:
        return (self.di.get_value(self._analysis_address("Y.GAUSS_MEAN")),
                self.di.get_value(self._analysis_address("Y.GAUSS_MEAN_ERR")))

    def get_rms_mean_x(self) -> tuple[float, float]:
        return (self.di.get_value(self._analysis_address("X.RMS_MEAN")),
                self.di.get_value(self._analysis_address("X.RMS_MEAN_ERR")))
        
    def get_gauss_mean_y(self) -> tuple[float, float]:
        return (self.di.get_value(self._analysis_address("Y.RMS_MEAN")),
                self.di.get_value(self._analysis_address("Y.RMS_MEAN_ERR")))
    
    def get_slices_gauss_sigma(self) -> tuple[float, float]:
        return (self.di.get_value(self._analysis_address("SLICES.GAUSS_SIGMA")),
                self.di.get_value(self._analysis_address("SLICES.GAUSS_SIGMA_ERR")))

    def get_last_sampled_image_dir(self) -> str:
        dirname = self.di.get_value(self._analysis_address("SAMPLES.TIMESTAMP"))
        node_name = self._get_node_name()
        return f"/doocs/{node_name}/server/imageanalysis_server/StoredData/{self.screen_name}/{dirname}"

    def _get_node_name(self) -> str:
        doocs_address = self.di.get_value("XFEL.DIAG/IMAGEANALYSIS/OTRC.64.I1D/SVR.ADDR")
        # property of value read by above address is what we want.
        return DOOCSAddress.from_string(doocs_address).property