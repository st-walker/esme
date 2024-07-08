from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import sys 
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QMainWindow

from PyQt5.QtCore import QAbstractTableModel, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QKeyEvent
from PyQt5.QtWidgets import QApplication, QHeaderView, QTableView, QVBoxLayout, QWidget, QMainWindow
from esme.control.screens import Screen, Position
from esme.control.exceptions import DOOCSReadError
from esme.gui.ui.calibrator import Ui_calibrator_mainwindow
import datetime
from pathlib import Path
import toml

from esme.core import DiagnosticRegion, region_from_screen_name
from esme.gui.widgets.common import get_machine_manager_factory, get_tds_calibration_config_dir, set_machine_by_region
from esme.gui.widgets.sbunchpanel import IBFBWarningDialogue

class IncompleteCalibration(RuntimeError):
    pass


@dataclass
class CalibrationContext:
    screen_name: str = ""
    snapshot: pd.DataFrame | None = None
    beam_energy: float = 0.0
    frequency: float = 0.0
    screen_position: Position = field(default_factory=lambda: Position.UNKNOWN)
    off_axis_roi_bounds: tuple[tuple[int, int], tuple[int, int]] | None = None
    pixel_sizes: tuple[float, float] = (np.nan, np.nan)

    def r12_streaking(self) -> float:
        # TODO: Make this make sense.
        return self.amplitude * self.frequency  # Example implementation


@dataclass
class PhaseScan:
    """Phase scan. This is output from the calibration routine."""

    prange: tuple[float | None, float | None] = (None, None)
    images: npt.NDArray | None = None

    def phases(self, n=5):
        return np.linspace(*self.prange, num=n)

    def coms(self):
        return self.images.sum(axis=1)
    
    def mean_phase(self) -> float:
        return 0.5 * (self.prange[0] + self.prange[1])

    def cal(self) -> float:
        try:
            phases = self.phases()
            coms = self.coms()
            return sum(phases * coms) # Place holder...
        except (np.AxisError, TypeError, AttributeError):
            raise IncompleteCalibration
    
    def has_real_bounds(self) -> bool:
        return isinstance(self.prange[0], float) and isinstance(self.prange[1], float)
    
    def voltage(self, context: CalibrationContext) -> float:
        return self.cal() * 10

@dataclass
class CalibrationSetpoint:
    """This is the combination of the input with its corresponding output."""

    amplitude: float | None = None
    pscan0: PhaseScan = field(default_factory=PhaseScan)
    pscan1: PhaseScan = field(default_factory=PhaseScan)
    background_images: npt.NDArray = 0.0

    def can_be_calibrated(self) -> bool:
        return all([self.amplitude is not None,
                    self.pscan0.has_real_bounds(),
                    self.pscan1.has_real_bounds()])
    
    def partially_defined_input(self) -> bool:
        return (any([self.amplitude is not None,
                    self.pscan0.has_real_bounds(),
                    self.pscan1.has_real_bounds()])
                    and not self.can_be_calibrated())

@dataclass
class TDSCalibration:
    context: CalibrationContext = field(default_factory=CalibrationContext)
    setpoints: List[CalibrationSetpoint] = field(
        default_factory=lambda: [CalibrationSetpoint() for _ in range(10)]
    )

    def v0(self) -> List[float]:
        pass

    def v1(self) -> List[float]:
        pass

    def vmean(self) -> float:
        pass


class CalibrationTableModel(QAbstractTableModel):
    def __init__(self, calibration: TDSCalibration | None = None):
        super().__init__()
        # Just have 5 rows, I don't see any need for more when calibrating...
        self.calibration = calibration or TDSCalibration([CalibrationSetpoint() for _ in range(5)])
        self.headers = [
            "Amplitude / %",
            "  𝜙₀₀ / °  ",
            "  𝜙₀₁ / °  ",
            "  𝜙₁₀ / °  ",
            "  𝜙₁₁ / °  ",
            "𝐶₀ / µmps⁻¹",
            "𝐶₁ / µmps⁻¹",
            "𝘝₀ / MV",
            "𝘝₁ / MV",
            "𝘝 / MV",
        ]

    def rowCount(self, parent=None):
        return len(self.calibration.setpoints)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        setpoint = self.calibration.setpoints[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return setpoint.amplitude
            elif index.column() == 1:
                return setpoint.pscan0.prange[0]
            elif index.column() == 2:
                return setpoint.pscan0.prange[1]
            elif index.column() == 3:
                return setpoint.pscan1.prange[0]
            elif index.column() == 4:
                return setpoint.pscan1.prange[1]
            
            elif index.column() == 5:
                # Calibration factor from first phase scan
                try:
                    return setpoint.pscan0.cal() 
                except IncompleteCalibration:
                    return ""
            elif index.column() == 6:
                # Calibration factor from other phase scan
                try:
                    return setpoint.pscan1.cal() 
                except IncompleteCalibration:
                    return ""
            elif index.column() == 7:
                # Derived voltage from first phase scan
                try:
                    return setpoint.pscan0.voltage(self.calibration.context)
                except IncompleteCalibration:
                    return ""
            elif index.column() == 8:
                # Derived voltage from second phase scan
                try:
                    return setpoint.pscan1.voltage(self.calibration.context)
                except IncompleteCalibration:
                    return ""
                # Average voltage.
            elif index.column() == 9:
                try:
                    v0 = setpoint.pscan0.voltage(self.calibration.context)
                    v1 = setpoint.pscan1.voltage(self.calibration.context)
                    return 0.5 * (v0 + v1)
                except IncompleteCalibration:
                    return ""

        if role == Qt.BackgroundRole:
            if index.column() in {1, 2, 5, 7}:
                color = QColor(255, 182, 193)  # Light salmon color
                color.setAlphaF(0.1)  # Set alpha to 0.1
                return QBrush(color)
            elif index.column() in {3, 4, 6, 8}:
                color = QColor(173, 216, 230)  # Baby blue color
                color.setAlphaF(0.1)  # Set alpha to 0.1
                return QBrush(color)

        return None
    
    def _try_and_derive_value(self, method):
        try:
            return method()
        except IncompleteCalibration:
            return ""    

    def _data_phase_ranges(self, setpoint, icol: int) -> float | None:
        assert 1 <= icol <= 4
        try:
            if icol == 1:
                return setpoint.pscan0.prange0[0]
            elif icol == 2:
                return setpoint.pscan0.prange0[1]
            elif icol == 3:
                return setpoint.pscan1.prange0[0]
            elif icol == 4:
                return setpoint.pscan1.prange0[1]
        except TypeError:
            return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        if index.row() >= len(self.calibration.setpoints):
            return False

        setpoint = self.calibration.setpoints[index.row()]

        try:
            value = float(value)
        except ValueError:
            return False
        icol = index.column()
        if icol == 0 and value >= 0:
            setpoint.amplitude = value or None
        elif 1 <= icol <= 4:
            self._set_data_phase_ranges(icol, setpoint, value)
        else:
            return False
        self.dataChanged.emit(index, index, [Qt.DisplayRole])
        return True
    
    def _set_data_phase_ranges(self, icol, setpoint, value):
        if icol == 1:
            prange = (value, setpoint.pscan0.prange[1])
            setpoint.pscan0.prange = prange
        elif icol == 2:
            prange = (setpoint.pscan0.prange[0], value)
            setpoint.pscan0.prange = prange
        elif icol == 3:
            prange = (value, setpoint.pscan1.prange[1])
            setpoint.pscan1.prange = prange
        elif icol == 4:
            prange = (setpoint.pscan1.prange[0], value)
            setpoint.pscan1.prange = prange

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        if index.column() in {0, 1, 2, 3, 4}:
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.headers[section]
        else:
            return section + 1


class TDSCalibratorMainWindow(QMainWindow):
    ORDERED_REGIONS = [DiagnosticRegion.I1, DiagnosticRegion.B2]
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_calibrator_mainwindow()
        self.ui.setupUi(self)
        # XXX: There should be two of these, one for I1 and one for B2!!!
        self.i1calibration = TDSCalibration()
        self.b2calibration = TDSCalibration()
        self._init_widget_stacks()
        self._connect_buttons()

        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine

        # We use this process pool for taking data (i.e. beam images)
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._executor.submit(lambda: None)

        # To handle IBFB warnings, which we shouldn't have switched on in conjunction with the fast kickers.
        self.ibfb_warning_dialogue = IBFBWarningDialogue(
            self.machine.sbunches, parent=self
        )
        self.ibfb_warning_dialogue.fire_signal.connect(
            lambda: self.machine.sbunches.start_diagnostic_bunch()
        )
        self.ibfb_warning_dialogue.disable_ibfb_aff_signal.connect(
            lambda: self.machine.sbunches.set_ibfb_lff(on=False)
        )

        self.ui.area_control.screen_name_signal.connect(self._set_screen)
        # Set initial screen name in the calibration context.
        self._set_screen(self.ui.area_control.get_selected_screen_name())
        self._update_calibration_context()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_calibration_parameters_ui)
        self.timer.start(1000)

    def _set_region(self, region: DiagnosticRegion) -> None:
        if region is DiagnosticRegion.I1:
            self.machine = self.i1machine
        elif region is DiagnosticRegion.B2:
            self.machine = self.b2machine

    def _get_active_region(self) -> DiagnosticRegion:
        if self.machine is self.i1machine:
            return DiagnosticRegion.I1
        elif self.machine is self.b2machine:
            return DiagnosticRegion.B2

    def _set_screen(self, screen_name: str) -> None:
        # there shoudl be some logic here, if a b2 screen, set screen for b2 calibration
        region = region_from_screen_name(screen_name)
        # If an i1 screen, set the screen for the i1 calibration instance, or if a b2
        # calibration then do it for the b2 instance.
        if region is DiagnosticRegion.I1:
            self.i1calibration.context.screen_name = screen_name
        elif region is DiagnosticRegion.B2:
            self.b2calibration.context.screen_name = screen_name
        # Set machine instance to be used.
        self._set_region(region)
        self._set_widget_stack_by_region(region)

    def _set_widget_stack_by_region(self, region: DiagnosticRegion) -> None:
        index = self.ORDERED_REGIONS.index(region)
        self.ui.table_stack.setCurrentIndex(index)
        self.ui.plot_stack.setCurrentIndex(index)
        self.ui.stackedWidget_2.setCurrentIndex(index)

    def _connect_buttons(self) -> None:
        self.ui.start_calibration_button.clicked.connect(self.calibrate_tds)
        self.ui.cancel_button.clicked.connect(self.interupt_calibration)
        self.ui.load_calibration_button.clicked.connect(self.load_calibration)

    def interupt_calibration(self) -> None:
        screen = self._get_screen()
        pos = screen.get_position()
        if pos is Position.OFFAXIS:
            self.machine.sbunches.stop_diagnostic_bunch()
        elif pos is Position.ONAXIS:
            self.machine.turn_beam_off()
        else:
            raise ValueError(f"Unknown screen position, {screen=}. {pos=}")

        self.set_ui_enabled(enabled=True)

    def _init_plots(self):
        pass

    def set_ui_enabled(self, *, enabled: bool) -> None:
        self.ui.area_control.setEnabled(enabled)
        self.ui.start_calibration_button.setEnabled(enabled)
        self.ui.cancel_button.setEnabled(not enabled)
        self.ui.load_calibration_button.setEnabled(enabled)
        self.ui.i1_calibration_table_view.setEnabled(enabled)
        self.ui.b2_calibration_table_view.setEnabled(enabled)

    def _update_calibration_parameters_ui(self, ctx: CalibrationContext) -> None:
        pass

    def _update_calibration_parameters_ui(self) -> None: 
        ctx = self._active_calibration()
        screen_name = ctx.screen_name
        optics_df = ctx.snapshot
        beam_energy = ctx.beam_energy
        screen_name = ctx.screen_name
        xsize, ysize = ctx.pixel_sizes
        if ctx.snapshot is not None:
            r12_streaking = self.machine.optics.r12_streaking_from_tds_to_point_from_df(optics_df,
                                                                                        screen_name,
                                                                                        beam_energy
                                                                                        )
            self.ui.i1_r12_streaking_value_label.setText(f"{r12_streaking:.2f} m/rad")
        else: 
            self.ui.i1_r12_streaking_label_value_label.setText(f"nan")

        self.ui.i1_beam_energy_value_label.setText(f"{beam_energy:.1f} MeV")
        self.ui.i1_tds_frequency_value_label.setText("3 GHz")
        self.ui.i1_screen_value_label.setText(f"{screen_name}")
        self.ui.i1_screen_position_value_label.setText(ctx.screen_position.name)
        self.ui.i1_pixel_size_value_label.setText(f"{ysize * 1e6:.2f} μm")

    def _write_calibration_info_box(self, prefix: str = "i1"):
        self.ui.i1_r12_streaking_value_label.setText(f"{r12_streaking:.2f} m/rad")
        self.ui.i1_r12_streaking_label_value_label.setText(f"nan")
        self.ui.i1_beam_energy_value_label.setText(f"{beam_energy:.1f} MeV")
        self.ui.i1_tds_frequency_value_label.setText("3 GHz")
        self.ui.i1_screen_value_label.setText(f"{screen_name}")
        self.ui.i1_screen_position_value_label.setText(ctx.screen_position.name)
        self.ui.i1_pixel_size_value_label.setText(f"{ysize * 1e6:.2f} μm")

        
        
    def _init_widget_stacks(self) -> None:
        # Init table views
        self._init_table_view(self.ui.i1_calibration_table_view, self.i1calibration)
        self._init_table_view(self.ui.b2_calibration_table_view, self.b2calibration)
        # Init the pyqtgraph graphics layout widgets

    def _active_calibration(self) -> TDSCalibration:
        region = self._get_active_region()
        if region is DiagnosticRegion.I1:
            return self.i1calibration
        elif region is DiagnosticRegion.B2:
            return self.b2calibration

    def _init_table_view(self, table_view, calibration: TDSCalibration) -> None:
        model = CalibrationTableModel(calibration)

        table_view.setModel(model)

        # # Set the last column to stretch and fill the remaining horizontal space
        header = table_view.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)  # Default for all columns
        header.setSectionResizeMode(9, QHeaderView.Stretch)  # Stretch the last column

    def _write_to_log(self, msg: str) -> None:
        self.ui.i1_log.append(msg)

    def _update_calibration_context(self) -> None:
        optics_df = self.machine.optics.optics_snapshot()
        screen = self._get_screen()
        screen_name = screen.name
        calibration = self._active_calibration()
        try:
            beam_energy = self.machine.optics.get_beam_energy()
        except DOOCSReadError:
            beam_energy = np.nan

        calibration.context.screen_name = screen_name
        calibration.context.snapshot = optics_df
        calibration.context.beam_energy = beam_energy
        try:
            xsize = screen.get_pixel_xsize()
            ysize = screen.get_pixel_ysize()
        except DOOCSReadError:
            xsize = ysize = np.nan

        calibration.context.pixel_sizes = xsize, ysize

        xminmax = screen.analysis.get_xroi_clipping()
        yminmax = screen.analysis.get_yroi_clipping()
        calibration.context.off_axis_roi_bounds = (xminmax, yminmax)

        self._update_calibration_parameters_ui()

    def _take_images(self, n: int) -> npt.NDArray:
        screen = self._get_screen()
        for _ in range(5):
            image_data_futures = self._executor.submit(_get_images_from_screen_instance, screen)

        images = []
        for future in as_completed(image_data_futures):
            images.append(future.result())

        return np.array(images)
    
    def _take_background(self) -> npt.NDArray:
        self._take_beam_off_screen()
        return self._take_images(n=5)

    def _turn_beam_onto_screen(self) -> None:
        screen = self._get_screen()
        position = screen.get_position()
        self._write_to_log(f"Screen: {screen.name}, position: {position}")
        if position is Position.ONAXIS:
            self.machine.turn_beam_on()
            # We clip the off-axis artefacts if the beam is on axis
            self.set_clipping(on=False)
        elif position is Position.OFFAXIS:
            self._kick_beam_onto_screen()
            # We are off axis so we enable clipping of off axis rubbish
            self.set_clipping(on=True)
        elif position is Position.OUT:
            # TODO: some sort of bombing here.
            pass

    def _take_beam_off_screen(self) -> None:
        screen = self._get_screen()
        position = screen.get_position()
        if position is Position.ONAXIS:
            self.machine.turn_beam_off()
            # We don't clip off-axis artefacts if the beam is on axis
            self.set_clipping(on=False)
        elif position is Position.OFFAXIS:
            self.machine.sbunches.stop_diagnostic_bunch()
            # We are off axis so we enable clipping of off axis rubbish
        elif position is Position.OUT:
            self._write_to_log("Camera")
            # TODO: some sort of bombing here.
            pass
        # Tidy up by disabling ROI clipping in the image analysis server
        # Maybe not really that important but just in case/nice to do.
        self.set_clipping(on=False)

    def _kick_beam_onto_screen(self) -> None:
        # Get the screen
        screen = self._get_screen()
        # Set the kickers for the screen
        self.machine.set_kickers_for_screen(screen.name)
        # Append a diagnostic bunch by setting bunch to last in machine + 1
        self.machine.sbunches.set_to_append_diagnostic_bunch()
        # Now start firing the fast kicker(s).
        self.safe_diagnostic_bunch_start()

    def safe_diagnostic_bunch_start(self) -> None:
        if self.machine.sbunches.is_either_ibfb_on():
            self.ibfb_warning_dialogue.show()
        else:
            self.machine.sbunches.start_diagnostic_bunch()

    def _get_screen(self) -> Screen:
        calibration = self._active_calibration()
        return self.machine.screens[calibration.context.screen_name]

    def calibrate_tds(self) -> None:
        """Main method that does the actual calibration, looping over the inputs in the table, 
        checking they're valid, and then doing the calibration at that setpoint
        (one setpoint = an amplitude and two pairs of phases)."""
        self.set_ui_enabled(enabled=False)
        self._update_calibration_context()
        calibrated_at_least_once = False
        # XXX: DO AUTO GAIN AT EVERY NEW STREAKING!!
        # XXX: AND ALSO TAKE NEW BACKGROUND!!
        for irow, setpoint in enumerate(self.calibration.setpoints, 1):            
            if setpoint.partially_defined_input():
                self._write_to_log(f"Skipping partially defined calibration input on row {irow}")
                continue
            if not setpoint.can_be_calibrated():
                # checking if line is simply blank in which case we just skip
                continue
            self.do_one_calibration_setpoint(irow, setpoint)
            done_at_least_one = True

        # reenable the UI now we are finished.
        self.set_ui_enabled(enabled=True)

        if calibrated_at_least_once:            
            self._write_to_log("Finished TDS Calibration")
            self.save_calibration()
        else:
            self._write_to_log("Not calibrating as no defined calibration setpoints")

    def do_one_calibration_setpoint(self, irow: int, setpoint: CalibrationSetpoint) -> None:
        # Set the TDS amplitude for this calibration setpoint
        # Go to the middle phase.
        # Turn on the beam.
        # Set auto gain
        # Take data
        # Turn the beam off
        # take background

        self._write_to_log(f"Starting calibration for row {irow} at {amp}")

        # Amplitude
        amp = setpoint.amplitude        
        self.machine.deflector.set_amplitude(amp)
        screen = self._get_screen()
        # Set central phase for the auto gain.
        self.machine.deflector.set_phase(setpoint.pscan.mean_phase())
        # Turn beam on.
        self._turn_beam_onto_screen()
        # Do auto gain for the screen with the beam on it.
        # We assume the ROI is correctly clipped whenever we are off-axis
        # as we take car of this in self._turn_beam_onto_screen()
        self._write_to_log("Setting camera gain for new amplitude...")
        screen.analysis.activate_gain_control()
        while screen.analysis.is_active(): # wait for gain adjustment to finish
            time.sleep(0.5)
        # Do first phase scan at first phase pair
        self._do_phase_scan(setpoint.pscan0)
        # Do second phase scan at the other phase pair
        self._do_phase_scan(setpoint.pscan1)

        # Take background (turns the beam off in the process)
        nbg = 5
        self._write_to_log(f"Taking {nbg} background images")
        bg_images = self._take_background(n=nbg)
        setpoint.background_images = bg_images

    def _do_phase_scan(self, pscan: PhaseScan) -> None:
        screen = self._get_screen()
        images = []
        self._write_to_log(f"Starting phase scan between {pscan.prange[0]} and {pscan.phrange[1]}")
        for i, phase in enumerate(pscan.phases()):
            if i == 0: # Sleep extra for the first step as it may involve a large jump in phase
                time.sleep(2.)
            self._write_to_log(f"Setting TDS phase to {phase}")
            self.machine.deflector.set_phase(phase)
            time.sleep(0.1)
            images = self._take_images(n=10)

        pscan.images = images

    def save_calibration(self) -> None:
        # XXX: Offer to save here with a qdialog...
        # First save the calibration context
        outdir = self._get_outdir()
        self._save_calibration_context(outdir)
        self._save_calibration_setpoints(outdir)
        self._write_to_log(f"Written calibration to {outdir}")

    def _get_outdir(self) -> Path:
        # Something sensible should go here...
        nowstr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        # XXX: Get rid of hardcoded "i1" please...
        return get_tds_calibration_config_dir() / "i1" / nowstr

    def _save_calibration_setpoints(self, outdir: Path) -> None:
        for i, setpoint in enumerate(self.calibration.setpoints):
            np.savez_compressed(outdir / f"setpoint-{i}.npz",
                                amplitude=setpoint.amplitude,
                                prange0=setpoint.pscan0.prange, 
                                images0=setpoint.pscan0.images,
                                prange1=setpoint.pscan1.prange,
                                images1=setpoint.pscan1.images,
                                bg_images=setpoint.background_images)
        
    def _save_calibration_context(self, outdir: Path) -> None:
        ctx = self.calibration.context
        ctx.snapshot.to_csv(outdir / "snapshot.csv")
        misc_context = {"screen_name": ctx.screen_name,
                        "beam_energy": ctx.beam_energy,
                        "tds_frequency": ctx.frequency,
                        "screen_position": ctx.screen_position.name,
                        "pixel_sizes": ctx.pixel_sizes,
                        "off_axis_roi_bounds": ctx.off_axis_roi_bounds}
        with (outdir / "context.toml").open("w") as f:
            toml.dump(misc_context, f)

    def load_calibration(self, region: DiagnosticRegion, caldir: Path):
        pass


def load_calibration_context(caldir):
    bg_images = np.load(caldir / "background.npz")

def _get_images_from_screen_instance(screen: Screen) -> dict[str, Any]:
    return screen.get_image_raw_full()

def start_tds_calibrator(argv) -> None:
    app = QApplication(argv)

    calwindow = TDSCalibratorMainWindow()
    calwindow.setWindowTitle("TDS Calibrator")
    calwindow.show()
    calwindow.raise_()
    sys.exit(app.exec_())

def create_sample_data():
    return TDSCalibration()

if __name__ == "__main__":
    app = QApplication([])

    calibration_data = create_sample_data()
    model = CalibrationTableModel(calibration_data)

    custom_view = TDSCalibratorMainWindow(model)

    window = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(custom_view)
    window.setLayout(layout)
    window.show()

    app.exec()
