from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
import sys
from copy import deepcopy
import time
import queue
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt, pyqtSignal, QObject, QThread, QThreadPool, QRunnable, QMutex, QMutexLocker
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QMainWindow, QFileDialog, QMessageBox
from threading import Event
from PyQt5.QtCore import QAbstractTableModel, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QKeyEvent
from PyQt5.QtWidgets import QApplication, QHeaderView, QTableView, QVBoxLayout, QWidget, QMainWindow
from esme.control.screens import Screen, Position
from esme.control.exceptions import DOOCSReadError, DOOCSWriteError
from esme.gui.ui.calibrator import Ui_calibrator_mainwindow
import os
import datetime
from pathlib import Path
import toml
from types import SimpleNamespace
import pyqtgraph as pg
import scipy.ndimage as ndi
from esme.calibration import calculate_voltage
import matplotlib.pyplot as plt
import toml
import numbers
from esme.image import zero_off_axis_regions
from esme.maths import get_gaussian_fit


from esme.core import DiagnosticRegion, region_from_screen_name
from esme.gui.widgets.common import get_machine_manager_factory, get_tds_calibration_config_dir, set_machine_by_region, send_widget_to_log, df_to_logbook_table, send_to_logbook
from esme.gui.widgets.sbunchpanel import IBFBWarningDialogue
from esme.control.exceptions import EuXFELMachineError
from esme.control.tds import StreakingPlane
from esme.maths import linear_fit
from uncertainties import ufloat
from esme.optics import calculate_i1d_r34_from_tds_centre
from functools import cached_property
from IPython import embed
from scipy import ndimage

_DEFAULT_COLOUR_CYCLE = plt.rcParams['axes.prop_cycle'].by_key()['color']


class IncompleteCalibration(RuntimeError):
    pass


class InterruptedCalibration(RuntimeError):
    pass

class CalibrationError(RuntimeError):
    def __init__(self, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message = message

@dataclass
class MeasurementContext:
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


@dataclass
class PhaseScan:
    """Phase scan. This is output from the calibration routine."""

    prange: tuple[float | None, float | None] = (None, None)
    # There should be len(self.phases(...)) sets of m images.
    images: npt.NDArray | None = None
    bg_images: npt.NDArray | None = None

    _calibration_fit: tuple[float, float] | None = None
    
    def phases(self, n: int = 5):
        return np.linspace(*self.prange, num=n)

    def centres_of_mass(self, ctx: MeasurementContext) -> list[ufloat]:
        mean_centres_of_mass = []
        mean_bg = self.bg_images.mean(axis=0)

        if ctx.streaking_plane is StreakingPlane.HORIZONTAL:
            psize = ctx.pixel_sizes[0]
        elif ctx.streaking_plane is StreakingPlane.VERTICAL:
            psize = ctx.pixel_sizes[1]
        else:
            raise TypeError()
        
        for image_collection_at_one_phase in self.images:
            this_phases_mean_beam_positions = []
            for image in image_collection_at_one_phase:
                image = self._do_image_analysis(ctx, image, mean_bg=mean_bg)
                # from IPython import embed; embed()
                timeproj = image.sum(axis=0) # ???
                # from IPython import embed; embed()
                popt, _ = get_gaussian_fit(np.arange(len(timeproj)), timeproj, p0=(timeproj.max(), timeproj.argmax(), 300))
                _, mu, _ = popt
                print(mu)
                this_phases_mean_beam_positions.append(ufloat(mu * psize, psize / 2))

            mean_centres_of_mass.append(np.mean(this_phases_mean_beam_positions, axis=1))

        return mean_centres_of_mass
    
    def _do_image_analysis(self, ctx: MeasurementContext, image: npt.NDArray, mean_bg: npt.NDArray) -> npt.NDArray:
        # Get rid of any off axis artefacts like the screen edge.
        if ctx.screen_position is Position.OFFAXIS:
            (xmin, xmax), (ymin, ymax) = ctx.off_axis_roi_bounds
            # zero_off_axis_regions(image, xmin, xmax, ymin, ymax)
            zero_off_axis_regions(image, ymin, ymax, xmin, xmax)
        # Subtract background and clip negative values.
        image = (image - mean_bg).clip(min=0)
        Hg = ndimage.gaussian_filter(image, sigma=5)
        threshold = 0.05
        inds_mask_neg = np.asarray((Hg - np.max(Hg) * threshold) < 0).nonzero()
        image[inds_mask_neg] = 0.0
        return image

    def mean_phase(self) -> float:
        return 0.5 * (self.prange[0] + self.prange[1])
    
    def phase_deltas(self, n: int = 5) -> float:
        return self.phases(n=n) - self.mean_phase()
    
    def time_deltas(self, ctx: MeasurementContext, n: int = 5) -> npt.NDArray:
        frequency = ctx.frequency
        phase_deltas = self.phase_deltas(n=n)
        return phase_deltas / (360 * frequency)

    def calibration_fit(self, ctx: MeasurementContext, n: int = 5) -> tuple[ufloat, ufloat]:
        if self._calibration_fit is not None:
            return self._calibration_fit
        phase_deltas = self.phase_deltas(n=n)
        coms = self.centres_of_mass(ctx=ctx)
        com_values = [c.n for c in coms]
        com_errors = [c.s for c in coms]
        yint, grad = linear_fit(phase_deltas, com_values, com_errors)
        yint = ufloat(*yint)
        grad = ufloat(*grad)
        # is grad here going to be a tuple of numbers, one 
        # This calibration is dy/dphi, not dy/dt..!

        return yint, grad
    
    def calibration_fit_time(self, ctx: MeasurementContext, n: int = 5) -> tuple[ufloat, ufloat]:
        yint, grad = self.calibration_fit(ctx, n=n)
        return yint, grad * 360 * ctx.frequency

    def cal(self, ctx: MeasurementContext) -> ufloat:
        try:
            _, grad = self.calibration_fit(ctx)
            return grad
        except (np.AxisError, TypeError, AttributeError):
            raise IncompleteCalibration
    
    def time_cal(self, ctx: MeasurementContext) -> ufloat:
        phase_cal = self.cal(ctx)
        time_cal = phase_cal * 360 * ctx.frequency
        return time_cal
    
    def has_real_bounds(self) -> bool:
        return isinstance(self.prange[0], numbers.Real) and isinstance(self.prange[1], numbers.Real)
    
    def bounds_are_distinct(self) -> bool:
        low = self.prange[0]
        hi = self.prange[1]
        if low is None or hi is None:
            return False
        return not np.isclose(low, hi)
    
    def voltage(self, ctx: MeasurementContext) -> ufloat:
        return calculate_voltage(slope=self.time_cal(ctx=ctx),
                                 r34=ctx.r12_streaking,
                                 energy=ctx.beam_energy,
                                 frequency=ctx.frequency)
    
    def is_complete(self) -> bool:
        return all([self.bounds_are_distinct(),
                    self.has_real_bounds(),
                    self.images is not None,
                    self.bg_images is not None])
    
    def clear_measurement(self) -> None:
        self.images = None
        self.bg_images = None


@dataclass
class CalibrationSetpoint:
    """This is the combination of the input with its corresponding output."""

    amplitude: float | None = None
    pscan0: PhaseScan = field(default_factory=PhaseScan)
    pscan1: PhaseScan = field(default_factory=PhaseScan)

    def can_be_calibrated(self) -> bool:
        return all([self.amplitude is not None,
                    self.pscan0.has_real_bounds(),
                    self.pscan0.bounds_are_distinct(),
                    self.pscan1.has_real_bounds(),
                    self.pscan1.bounds_are_distinct()])
    
    def partially_defined_input(self) -> bool:
        inputs_are_defined = [self.amplitude is not None,
                                self.pscan0.has_real_bounds(),
                                self.pscan1.has_real_bounds()]
        return any(inputs_are_defined) and not all(inputs_are_defined)
    
    def vmean(self, ctx: MeasurementContext) -> ufloat:
        v0 = self.pscan0.voltage(ctx=ctx)
        v1 = self.pscan1.voltage(ctx=ctx)
        return (abs(v0) + abs(v1)) / 2.0
    
    def is_complete(self) -> bool:
        return self.pscan0.is_complete() and self.pscan1.is_complete()
    
    def clear_measurement(self):
        self.pscan0.clear_measurement()
        self.pscan1.clear_measurement()

    

@dataclass
class TDSCalibration:
    context: MeasurementContext = field(default_factory=MeasurementContext)
    setpoints: List[CalibrationSetpoint] = field(
        default_factory=lambda: [CalibrationSetpoint() for _ in range(len(_DEFAULT_COLOUR_CYCLE))]
    )

    def amplitudes(self) -> list[float]:
        result = []
        for sp in self.setpoints:
            if not sp.is_complete():
                continue
            result.append(sp.amplitude)
        return result

    def vmean(self) -> list[ufloat]:
        return [sp.vmean(ctx=self.context) for sp in self.setpoints if sp.is_complete()]


class CalibrationTableModel(QAbstractTableModel):
    def __init__(self, calibration: TDSCalibration | None = None):
        super().__init__()
        # Just have 5 rows, I don't see any need for more when calibrating...
        self.calibration = calibration or TDSCalibration([CalibrationSetpoint() for _ in range(5)])
        # XXX: Hardcoded for debugging for now
        sp0 = self.calibration.setpoints[0]
        sp0.amplitude = 26
        sp0.pscan0.prange = (-63, -45)
        sp0.pscan1.prange = (-243, -226)
        
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

    def update_data(self):
        self.beginResetModel()
        self.endResetModel()

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
                print(setpoint.pscan0.prange)
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
                    # Convert from m/s to µm/ps.
                    result = setpoint.pscan0.time_cal(self.calibration.context).n * 1e-6
                    # print(result)
                    return result
                except IncompleteCalibration:
                    return ""
            elif index.column() == 6:
                # Calibration factor from other phase scan
                try:
                    # Convert from m/s to µm/ps.
                    result = setpoint.pscan1.time_cal(self.calibration.context).n * 1e-6
                    return result
                except IncompleteCalibration:
                    return ""
            elif index.column() == 7:
                # Derived voltage from first phase scan
                try:
                    # convert to MV
                    return setpoint.pscan0.voltage(self.calibration.context).n / 1e6
                except IncompleteCalibration:
                    return ""
            elif index.column() == 8:
                # Derived voltage from second phase scan
                try:
                    # Conver to MV
                    return setpoint.pscan1.voltage(self.calibration.context).n / 1e6
                except IncompleteCalibration:
                    return ""
                # Average voltage.
            elif index.column() == 9:
                try: 
                    # Conver to MV
                    return setpoint.vmean(self.calibration.context).n / 1e6
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
        
    def to_df(self) -> pd.DataFrame:
        rows = self.rowCount()
        columns = self.columnCount()

        data = []
        for row in range(rows):
            row_data = []
            for column in range(columns):
                index = self.index(row, column)
                row_data.append(self.data(index))
            data.append(row_data)

        df = pd.DataFrame(data)

        headers = [self.headerData(col, Qt.Horizontal) for col in range(columns)]
        df.columns = headers
        
        return df

        
class CalibrationWorkerSignals(QObject):
    # For incremental updates we post a completed setpoint
    finished_calibration_setpoint = pyqtSignal(int, CalibrationSetpoint)
    # We have completely finished the calibration
    finished_calibration = pyqtSignal()
    # For any helpful printing.
    log_message = pyqtSignal(str)
    # offer to save the calibration that we have successfully completed
    save_calibration = pyqtSignal()
    # fatal exceptions to pass to parent thread
    calibration_error = pyqtSignal(CalibrationError)


class TDSCalibrationWorker(QRunnable):
    def __init__(self, machine_interface, screen, calibration) -> None:
        super().__init__()

        self.screen = screen
        self.machine = machine_interface
        self.calibration = calibration

        self.signals = CalibrationWorkerSignals()

        self.interrupt_event = Event()

    def _write_to_log(self, msg: str) -> None:
        print(msg)
        self.signals.log_message.emit(msg)

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_event.is_set():
            raise InterruptedCalibration()

    def run(self) -> None:
        try:
            self._calibrate_tds()
        except InterruptedCalibration:
            pass
        except DOOCSWriteError as e:
            message = f"Unable to write {e.value} to {e.address}"
            self.signals.calibration_error.emit(CalibrationError(message))

        print("We have finished...")
        self.signals.finished_calibration.emit()

    def _calibrate_tds(self) -> None:
        calibrated_at_least_once = False
        print("calibrating...")
        for irow, setpoint in enumerate(self.calibration.setpoints, 1):
            if setpoint.partially_defined_input():
                self._write_to_log(f"Skipping partially defined calibration input on row {irow}")
                continue
            if not setpoint.can_be_calibrated():
                # checking if line is simply blank in which case we just skip
                continue
            try:
                setpoint = self.do_one_calibration_setpoint(irow, setpoint)
                self.signals.finished_calibration_setpoint.emit(irow, setpoint)
            except EuXFELMachineError:
                break
            calibrated_at_least_once = True

        if calibrated_at_least_once:            
            self._write_to_log("Finished TDS Calibration")
            self.signals.save_calibration.emit()
        else:
            self._write_to_log("Not calibrating as no defined calibration setpoints")

    def do_one_calibration_setpoint(self, irow: int, setpoint: CalibrationSetpoint) -> CalibrationSetpoint:
        # Set the TDS amplitude for this calibration setpoint
        # Go to the middle phase.
        # Turn on the beam.
        # Set auto gain
        # Take data
        # Turn the beam off
        # take background

        # deepcopy to keep things simple.
        setpoint = deepcopy(setpoint)

        # Amplitude
        amp = setpoint.amplitude
        self.machine.tds.set_amplitude(amp)
        self._write_to_log(f"Starting calibration for row {irow} at {amp}")
        # Set central phase for the auto gain.  arbitrarily do it with middle of first pscan phase range.
        self.machine.tds.set_phase(setpoint.pscan0.mean_phase())
        # Turn beam on.
        self.machine.turn_beam_onto_screen(self.screen, streak=True)
        # Do auto gain for the screen with the beam on it.
        # We assume the ROI is correctly clipped whenever we are off-axis
        # as we take care of this in self._turn_beam_onto_screen()
        self._write_to_log("Setting camera gain for new amplitude...")
        self.screen.analysis.activate_gain_control()        
        while self.screen.analysis.is_active(): # wait for gain adjustment to finish
            self._raise_if_interrupted()
            time.sleep(0.5)
        time.sleep(4)
        # Do first phase scan at first phase pair
        self._do_phase_scan(setpoint.pscan0)
        # Do second phase scan at the other phase pair
        self._do_phase_scan(setpoint.pscan1)

        # Take background (turns the beam off in the process)
        nbg = 5
        self._write_to_log(f"Taking {nbg} background images")
        bg_images = self._take_background(n=nbg)
        setpoint.pscan0.bg_images = bg_images
        setpoint.pscan1.bg_images = bg_images

        return setpoint

    def _do_phase_scan(self, pscan: PhaseScan) -> None:
        images = []
        self._write_to_log(f"Starting phase scan between {pscan.prange[0]} and {pscan.prange[1]}")
        for i, phase in enumerate(pscan.phases()):
            self.machine.tds.set_phase(phase)
            if i == 0: # Sleep extra for the first step as it may involve a large jump in phase
                time.sleep(2.)
            self._write_to_log(f"Setting TDS phase to {phase}")
            time.sleep(1)
            images.append(self._take_images(n=10))

        pscan.images = images

    def _take_images(self, n: int) -> npt.NDArray:
        return np.array([self.screen.get_image_raw().T.astype(np.float64) for _ in range(n)])
    
    def _take_background(self, n: int) -> npt.NDArray:
        # Turn beam off screen.
        # Then take data
        self.machine.take_beam_off_screen(self.screen)
        return self._take_images(n=n)
    
class PhaseCheckerWorker(TDSCalibrationWorker):
    def run(self):
        try:
            self._check_phases()
        except InterruptedCalibration:
            return

        self.signals.finished_calibration()

    def _check_phases(self):
        for irow, setpoint in enumerate(self.calibration.setpoints, 1):
            if setpoint.partially_defined_input():
                self.signals.log_message.emit(f"Skipping checking partially defined calibration input on row {irow}")
                continue
            if not setpoint.can_be_calibrated():
                # checking if line is simply blank in which case we just skip
                continue
            try:
                amplitude = setpoint.amplitude
                self.signals.log_message(f"Set TDS amplitude {amplitude}%")
                self.machine.tds.set_amplitude(amplitude)
                self._raise_if_interrupted()
                time.sleep(2)
                self._check_prange(setpoint.pscan0.prange)
                self._check_prange(setpoint.pscan1.prange)
            except EuXFELMachineError:
                break

    def _check_prange(self, prange: tuple[float, float]) -> None:
        phase0, phase1 = prange
        self.signals.log_message(f"Set phase: {phase0}°")
        self.machine.tds.set_phase(phase0)
        self._raise_if_interrupted()
        time.sleep(2)
        self.signals.log_message(f"Set phase: {phase1}°")
        self.machine.tds.set_phase(phase1)
        self._raise_if_interrupted()
        time.sleep(2)

class TDSCalibratorMainWindow(QMainWindow):
    # I need to do this in another thread...  even though I don't want to 😱
    ORDERED_REGIONS = [DiagnosticRegion.I1, DiagnosticRegion.B2]
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_calibrator_mainwindow()
        self.ui.setupUi(self)
        self.i1machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.I1)
        self.b2machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.B2)
        self.machine = self.i1machine
        
        # These are passed to the CalibrationTableView in _init_table_view and form the underlying data model.
        self.i1calibration, self.b2calibration = self._load_most_recent_calibrations()
        # ???

        self._init_widget_stacks()
        self._connect_buttons()

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

        self.plots = SimpleNamespace()
        self._init_plots()

        self.threadpool = QThreadPool()
        self.worker = None

        self.ui.area_control.screen_name_signal.connect(self._set_screen)
        # Set initial screen name in the calibration context.
        self._set_screen(self.ui.area_control.get_selected_screen_name())
        self._update_calibration_context()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_calibration_parameters_ui)
        self.timer.start(1000)

    def _handle_errors(self, exception: CalibrationError) -> None:
        msg = exception.message
        result = QMessageBox.critical(None, "Fatal Calibration Error",
                                     f"The calibration runner failed with the error: {msg}")

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
        self._update_calibration_context()
        self._set_region(region)
        self._set_widget_stack_by_region(region)
        self._update_calibration_parameters_ui()

    def _set_widget_stack_by_region(self, region: DiagnosticRegion) -> None:
        index = self.ORDERED_REGIONS.index(region)
        self.ui.table_stack.setCurrentIndex(index)
        self.ui.plot_stack.setCurrentIndex(index)

    def _connect_buttons(self) -> None:
        self.ui.start_calibration_button.clicked.connect(self.calibrate_tds)
        self.ui.cancel_button.clicked.connect(self.interrupt_calibration)
        self.ui.load_calibration_button.clicked.connect(self.load_calibration)
        self.ui.check_phases_button.clicked.connect(self.check_phases)

    def check_phases(self) -> None:
        self.worker = PhaseCheckerWorker(self.machine, self._get_screen(), self._active_calibration())
        self.worker.signals.log_message.connect(self._write_to_log)
        self.worker.signals.finished_calibration.connect(lambda: self.set_ui_enabled(enabled=True))
        self.worker.signals.calibration_error.connect(self._handle_errors)
        # self.threadpool.start(self.worker)
        self.set_ui_enabled(enabled=False)
        self.worker.run()

    def interrupt_calibration(self) -> None:
        if self.worker is None:
            return
        self.worker.interrupt_event.set()
        self.i1machine.sbunches.stop_diagnostic_bunch()
        self.b2machine.sbunches.stop_diagnostic_bunch()
        # And also turn the beam off if we are on-axis.
        if self._get_screen().get_position() is Position.ONAXIS:
            self.machine.turn_beam_off()
        self.set_ui_enabled(enabled=True)

    def _init_plots(self) -> None:
        self.plots.i1_voltage_plot = self.ui.i1_calibration_graphics.addPlot(title="Voltage Calibration")
        self.plots.b2_voltage_plot = self.ui.b2_calibration_graphics.addPlot(title="Voltage Calibration")
        self._init_voltage_calibration_plot(self.plots.i1_voltage_plot)
        self._init_voltage_calibration_plot(self.plots.b2_voltage_plot)

        self.plots.i1_pscan_plot = self.ui.i1_pscan_graphics.addPlot(title="Phase Scan")
        self.plots.b2_pscan_plot = self.ui.b2_pscan_graphics.addPlot(title="Phase Scan")
        self._init_calibration_plot(self.plots.i1_pscan_plot)
        self._init_calibration_plot(self.plots.b2_pscan_plot)

    def _init_voltage_calibration_plot(self, plot_item: pg.PlotItem) -> None:
        plot_item.setLabel('bottom', 'TDS Amplitude', units="%")
        plot_item.setLabel("left", "Voltage", units="V")

    def _plot_phase_scans(self) -> None:
        self._plot_phase_scan(self.plots.i1_pscan_plot, self.i1calibration)
        self._plot_phase_scan(self.plots.b2_pscan_plot, self.b2calibration)

    def _plot_phase_scan(self, plot_item: pg.PlotItem, calibration: TDSCalibration) -> None:
        ctx = calibration.context
        for i, setpoint in enumerate(calibration.setpoints):
            amp = setpoint.amplitude
            coms0 = setpoint.pscan0.centres_of_mass(ctx)
            coms1 = setpoint.pscan1.centres_of_mass(ctx)
            delta_time0 = setpoint.pscan0.time_deltas(ctx)
            delta_time1 = setpoint.pscan1.time_deltas(ctx)
            errors0 = [dt.s/2. for dt in delta_time0]
            errors1 = [dt.s/2. for dt in delta_time1]

            error_bar_item0 = pg.ErrorBarItem(x=delta_time0,
                                              y=[dt.n for dt in delta_time0],
                                              top=errors0,
                                              bottom=errors0,
                                              beam=0.1,
                                              pen=_DEFAULT_COLOUR_CYCLE[i])

            error_bar_item1 = pg.ErrorBarItem(x=delta_time0,
                                              y=[dt.n for dt in delta_time0],
                                              top=errors1,
                                              bottom=errors1,
                                              beam=0.1,
                                              pen=_DEFAULT_COLOUR_CYCLE[i])

            plot_item.addItem(error_bar_item0)
            plot_item.addItem(error_bar_item1)

    def _plot_calibrations(self) -> None:
        self._plot_calibration(self.plots.i1_voltage_plot, self.i1calibration)
        self._plot_calibration(self.plots.b2_voltage_plot, self.b2calibration)
    
    def _plot_calibration(self, plot_item: pg.PlotItem, calibration: TDSCalibration) -> None:
        plot_item.clear()
        ctx = calibration.context
        amplitudes = calibration.amplitudes
        voltages = calibration.vmean(ctx=ctx)
        voltage_values = [v.n for v in voltages]
        voltage_errors = [v.s/2 for v in voltages]
        error_bar_item = pg.ErrorBarItem(x=amplitudes,
                                         y=voltage_values,
                                         top=voltage_errors,
                                         bottom=voltage_errors,
                                         beam=0.1)
        plot_item.addItem(error_bar_item)

    def _init_calibration_plot(self, plot_item: pg.PlotItem) -> None:
        plot_item.setLabel('bottom', 'Δ𝑡', units="s")
        plot_item.setLabel('left', 'Centre of Mass', units="m")
        plot_item.addLegend()

    def set_ui_enabled(self, *, enabled: bool) -> None:
        self.ui.area_control.setEnabled(enabled)
        self.ui.start_calibration_button.setEnabled(enabled)
        self.ui.cancel_button.setEnabled(not enabled)
        self.ui.load_calibration_button.setEnabled(enabled)
        self.ui.i1_calibration_table_view.setEnabled(enabled)
        self.ui.b2_calibration_table_view.setEnabled(enabled)

    def _update_calibration_parameters_ui(self) -> None: 
        ctx = self._active_calibration().context
        screen_name = ctx.screen_name
        optics_df = ctx.snapshot
        beam_energy = ctx.beam_energy
        screen_name = ctx.screen_name
        xsize, ysize = ctx.pixel_sizes
        frequency = ctx.frequency
        if ctx.snapshot is not None:
            r12_streaking = self.machine.optics.r12_streaking_from_tds_to_point_from_df(optics_df,
                                                                                        screen_name,
                                                                                        beam_energy)
                                                                                        
            self.ui.r12_streaking_value_label.setText(f"{r12_streaking:.2f} m/rad")
        else: 
            self.ui.r12_streaking_value_label.setText(f"nan")

        self.ui.beam_energy_value_label.setText(f"{beam_energy:.1f} MeV")
        self.ui.tds_frequency_value_label.setText(f"{frequency/1e9:.0g} GHz")
        self.ui.screen_value_label.setText(f"{screen_name}")
        self.ui.screen_position_value_label.setText(ctx.screen_position.name)
        self.ui.pixel_size_value_label.setText(f"{ysize * 1e6:.2f} μm")

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
        if self._get_active_region() is DiagnosticRegion.I1:
            self.ui.i1_log.append(msg)
        elif self._get_active_region() is DiagnosticRegion.B2:
            self.ui.b2_log_append(msg)

    def _update_calibration_context(self) -> None:
        optics_df = self.machine.optics.optics_snapshot()
        screen = self._get_screen()
        screen_name = screen.name
        calibration = self._active_calibration()
        try:
            beam_energy = self.machine.optics.get_beam_energy()
        except DOOCSReadError: # XXX: THIS IS NAUGHTY!!
            beam_energy = 130

        calibration.context.screen_name = screen_name
        calibration.context.snapshot = optics_df
        calibration.context.beam_energy = beam_energy
        # XXX: hardcoded, should be in tds.py class??
        calibration.context.frequency = 3e9
        calibration.context.streaking_plane = self.machine.tds.plane
        try:
            xsize = screen.get_pixel_xsize()
            ysize = screen.get_pixel_ysize()
        except DOOCSReadError:
            xsize = ysize = np.nan

        calibration.context.pixel_sizes = xsize, ysize
        calibration.context.screen_position = screen.get_position()

        xminmax = screen.analysis.get_xroi_clipping()
        yminmax = screen.analysis.get_yroi_clipping()
        calibration.context.off_axis_roi_bounds = (xminmax, yminmax)

        self._update_calibration_parameters_ui()

    def _get_screen(self) -> Screen:
        calibration = self._active_calibration()
        return self.machine.screens[calibration.context.screen_name]
    
    def _update_table_view(self):
        self.ui.i1_calibration_table_view.model().update_data()
        self.ui.b2_calibration_table_view.model().update_data()

    def _post_calibration_setpoint(self, index: int, setpoint: CalibrationSetpoint) -> None:
        # Update the corresponding entry.
        # We need to convert a visual row index (which start from 1) to the internal index
        # Which starts from 0.
        index -= 1
        cal = self._active_calibration()
        ctx = cal.context
        cal.setpoints[index].amplitude = setpoint.amplitude
        cal.setpoints[index].pscan0.bg_images = setpoint.pscan0.bg_images
        cal.setpoints[index].pscan1.bg_images = setpoint.pscan1.bg_images
        cal.setpoints[index].pscan0 = setpoint.pscan0
        cal.setpoints[index].pscan1 = setpoint.pscan1

        # sp = self.i1calibration.setpoints[0]
        # sp.pscan0.centres_of_mass(self.i1calibration.context)

        self._update_table_view()

        phase_deltas0 = setpoint.pscan0.time_deltas(ctx)
        phase_deltas1 = setpoint.pscan1.time_deltas(ctx)
        coms0 = setpoint.pscan0.centres_of_mass(ctx)
        coms1 = setpoint.pscan1.centres_of_mass(ctx)

        colour = _DEFAULT_COLOUR_CYCLE[index]
        pen = pg.mkPen(colour, width=4.5)
        self.plots.i1_pscan_plot.plot(phase_deltas0, [c.n for c in coms0], name=f"{setpoint.amplitude}%",
                                                    pen=pen)
        self.plots.i1_pscan_plot.plot(phase_deltas1, [c.n for c in coms1], pen=pen)

        errorbars0 = pg.ErrorBarItem(x=phase_deltas0, y=[c.s for c in coms0], pen=pen)
        errorbars1 = pg.ErrorBarItem(x=phase_deltas1, y=[c.s for c in coms1], pen=pen)

        self.plots.i1_pscan_plot.addItem(errorbars0)
        self.plots.i1_pscan_plot.addItem(errorbars1)


        # Now we should try to update the corresponding table view 

    def calibrate_tds(self) -> None:
        """Main method that does the actual calibration, looping over the inputs in the table, 
        checking they're valid, and then doing the calibration at that setpoint
        (one setpoint = an amplitude and two pairs of phases)."""
        self.worker = TDSCalibrationWorker(self.machine, self._get_screen(), self._active_calibration())
        self.worker.signals.log_message.connect(self._write_to_log)
        self.worker.signals.save_calibration.connect(self.save_calibration)
        self.worker.signals.finished_calibration_setpoint.connect(self._post_calibration_setpoint)
        self.worker.signals.finished_calibration.connect(lambda: self.set_ui_enabled(enabled=True))
        self.worker.signals.finished_calibration.connect(self._plot_calibration)
        self.worker.signals.calibration_error.connect(self._handle_errors)
        # self.threadpool.start(self.worker)
        self.set_ui_enabled(enabled=False)
        self.worker.run()

    def save_calibration(self) -> None:
        response = QMessageBox.question(None, "Save Calibration",
                                        "The calibration has finished, do you want to save it?", 
                                       QMessageBox.Yes | QMessageBox.No,
                                       defaultButton=QMessageBox.Yes)
        if response == QMessageBox.Yes:
            outdir = self._get_outdir()
            outdir.mkdir(exist_ok=True, parents=True)
            self._save_calibration_context(outdir)
            self._save_calibration_setpoints(outdir)
            self._save_avmapping(outdir)            
            self._write_to_log(f"Written calibration to {outdir}")

    def _write_calibration_info_to_logbook(self) -> None:
        region = self._get_active_region()
        if region is DiagnosticRegion.I1:
            df = self.ui.i1_calibration_table_view.model().to_df()
        elif region is DiagnosticRegion.B2:
            df = self.ui.b2_calibration_table_view.model().to_df()
        formatted_table_string = df_to_logbook_table(df)
        send_to_logbook(author="xfeloper",
                        title=f"{region.name} TDS Calibration", 
                        severity="MEASURE",
                        text=formatted_table_string,
                        image=None)

    def _save_avmapping(self, outdir: Path) -> None:
        # Save mapping of amplitudes to voltages.
        cal = self._active_calibration()
        amplitudes = [s.amplitude for s in cal.setpoints]
        voltages = [s.vmean(cal.context) for s in cal.setpoints]
        area = self._get_active_region()

        result = {"area": area.name, 
                  "amplitudes": amplitudes, 
                  "voltages": voltages}
        
        with (outdir / "avmapping.toml").open("w") as f:
            toml.dump(result, f)

    def _get_outdir(self) -> Path:
        # Something sensible should go here...
        nowstr = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        region = self._get_active_region().name.lower()
        return get_tds_calibration_config_dir() / region / nowstr

    def _save_calibration_setpoints(self, outdir: Path) -> None:
        calibration = self._active_calibration()
        for i, setpoint in enumerate(calibration.setpoints, 1):
            if setpoint.partially_defined_input():
                continue
            if not setpoint.can_be_calibrated():
                continue
            amp = setpoint.amplitude
            np.savez_compressed(outdir / f"setpoint-{i}-{amp}%.npz",
                                amplitude=setpoint.amplitude,
                                prange0=setpoint.pscan0.prange, 
                                images0=setpoint.pscan0.images,
                                prange1=setpoint.pscan1.prange,
                                images1=setpoint.pscan1.images,
                                bg_images=setpoint.pscan1.bg_images)
            
        
    def _save_calibration_context(self, outdir: Path) -> None:
        cal = self._active_calibration()
        save_measurement_context(outdir, cal.context)

    def _get_calibration_outdir(self, region: DiagnosticRegion) -> Path:
        region = self._get_active_region()
        rname = region.name
        outdir = get_tds_calibration_config_dir() / rname.lower()
        return outdir

    def load_calibration(self) -> None:
        region = self._get_active_region()
        rname = region.name
        outdir = get_tds_calibration_config_dir() / rname.lower()
        cal_dir = QFileDialog.getExistingDirectory(
            self, f"Load {rname} TDS Calibration", str(outdir))
        
        if not cal_dir:
            return
        
        calibration = load_calibration(Path(cal_dir))
        if region is DiagnosticRegion.I1:
            self.i1calibration.context = calibration.context
            self.i1calibration.setpoints = calibration.setpoints 
        elif region is DiagnosticRegion.B2:
            self.b2calibration.context = calibration.context
            self.b2calibration.setpoints = calibration.setpoints
        else:
            raise RuntimeError(region)
        
    def _load_most_recent_calibrations(self) -> tuple[TDSCalibration, TDSCalibration]:
        i1dir = self._get_calibration_outdir(DiagnosticRegion.I1)
        i1_calib_dirs = [d for d in i1dir.glob("*") if d.is_dir()]
        b2dir = self._get_calibration_outdir(DiagnosticRegion.B2)
        b2_calib_dirs = [d for d in i1dir.glob("*") if d.is_dir()]

        try:
            newest_i1cal_dir = next(iter(sorted(i1_calib_dirs, key=os.path.getmtime)))
        except StopIteration:
            i1calibration = TDSCalibration()
        else:
            try:
                i1calibration = load_calibration(newest_i1cal_dir)
            except FileNotFoundError:
                i1calibration =  TDSCalibration()
            else:
                self.ui.i1_log.append(f"Loading last I1 Calibration: {newest_i1cal_dir}")

        try:
            newest_b2cal_dir = next(iter(sorted(b2_calib_dirs, key=os.path.getmtime)))
        except StopIteration:
            b2calibration = TDSCalibration()
        else:
            try:
                b2calibration = load_calibration(newest_b2cal_dir)
            except FileNotFoundError:
                b2calibration = TDSCalibration()
            else:
                self.ui.b2_log.append(f"Loading last B2 Calibration: {newest_b2cal_dir}")

        return i1calibration, b2calibration


def save_measurement_context(outdir: str, ctx: MeasurementContext, **kwargs) -> None:
    ctx.snapshot.to_csv(outdir / "snapshot.csv")
    misc_context = {"screen_name": ctx.screen_name,
                    "beam_energy": ctx.beam_energy,
                    "tds_frequency": ctx.frequency,
                    "screen_position": ctx.screen_position.name,
                    "pixel_sizes": ctx.pixel_sizes,
                    "off_axis_roi_bounds": ctx.off_axis_roi_bounds,
                    "streaking_plane": ctx.streaking_plane.name}
    misc_context |= kwargs
    with (outdir / "context.toml").open("w") as f:
        toml.dump(misc_context, f)

def load_calibration(caldir: Path) -> TDSCalibration:
    calibration = TDSCalibration()
    calibration.context = load_calibration_context(caldir)
    calibration.setpoints = load_calibration_setpoints(caldir)
    return calibration

def load_calibration_setpoints(caldir: Path) -> list[CalibrationSetpoint]:
    setpoints = []
    for setpointnpz in caldir.glob("setpoint*.npz"):
        setpoints.append(load_calibration_setpoint(setpointnpz))
    return sorted(setpoints, key=lambda s: s.amplitude)

def load_calibration_setpoint(setpointnpz: Path) -> CalibrationSetpoint:
    arr = np.load(setpointnpz)
    amplitude = arr["amplitude"].item()
    prange0 = tuple(arr["prange0"])
    images0 = arr["images0"]
    prange1 = tuple(arr["prange1"])
    images1 = arr["images1"]
    bg_images = arr["bg_images"]

    pscan0 = PhaseScan(prange=prange0, images=images0, bg_images=bg_images)
    pscan1 = PhaseScan(prange=prange1, images=images1, bg_images=bg_images)

    return CalibrationSetpoint(amplitude=amplitude,
                               pscan0=pscan0,
                               pscan1=pscan1)


def load_calibration_context(caldir: Path) -> MeasurementContext:
    snapshot = pd.read_csv(caldir / "snapshot.csv")
    result = MeasurementContext()
    result.snapshot = snapshot
    with (caldir / "context.toml").open("r") as f:
        tdict = toml.load(f)


    result.screen_name = tdict["screen_name"]
    result.beam_energy = tdict["beam_energy"]
    result.frequency = tdict["tds_frequency"]
    result.screen_position = Position[tdict["screen_position"]]
    result.pixel_sizes = tdict["pixel_sizes"]
    result.off_axis_roi_bounds = tdict["off_axis_roi_bounds"]
    result.streaking_plane = StreakingPlane[tdict["streaking_plane"]]
    return result

def start_tds_calibrator(argv) -> None:
    app = QApplication(argv)

    calwindow = TDSCalibratorMainWindow()
    calwindow.setWindowTitle("TDS Calibrator")
    calwindow.show()
    calwindow.raise_()
    sys.exit(app.exec_())
