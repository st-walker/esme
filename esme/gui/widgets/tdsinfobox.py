from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal


from esme.gui.ui.calibration_infobox import Ui_Form

from dataclasses import dataclass


@dataclass
class TDSMeasurementContext:
    screen_name: str = ""
    snapshot: pd.DataFrame | None = None
    beam_energy: float = 0.0
    frequency: float = 0.0
    screen_position: Position = field(default_factory=lambda: Position.UNKNOWN)
    off_axis_roi_bounds: tuple[tuple[int, int], tuple[int, int]] | None = None
    pixel_sizes: tuple[float, float] = (np.nan, np.nan)
    streaking_plane: StreakingPlane = field(default_factory=lambda: StreakingPlane.UNKNOWN)

    @cached_property
    def r12_streaking(self) -> float:
        if region_from_screen_name(self.screen_name) is DiagnosticRegion.I1:
            return calculate_i1d_r34_from_tds_centre(self.snapshot, self.screen_name, self.beam_energy)
        else:
            raise ValueError()
        
    def get_streaked_pixel_size(self) -> float:
        if self.streaking_plane is Position.HORIZONTAL:
            return self.pixel_sizes[0]
        if self.streaking_plane is Position.VERTICAL:
            return self.pixel_sizes[1]
        raise TypeError()





class TDSMeasurementInfoBox(QWidget):
    voltage_calibration_signal = pyqtSignal(object)

    def __init__(self):
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.

    def set_screen(self, screen_name: str):
        pass

