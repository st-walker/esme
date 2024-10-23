import os
import sys
import time
from threading import Event
from dataclasses import dataclass, field
from uncertainties import ufloat
from textwrap import dedent

from PyQt5.QtWidgets import QMainWindow, QApplication, QDialogButtonBox, QVBoxLayout, QLabel, QDialog, QPushButton, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer, QRunnable, QObject, QThreadPool, pyqtSignal
import numpy as np
from scipy.optimize import curve_fit
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import pyqtSignal

from esme.gui.ui.quickcal import Ui_quickcal_window
from esme.gui.widgets.tdsmini import MiniTDS
from esme.control.machines import CalibrationManager
from esme.core import DiagnosticRegion, region_from_screen_name
from esme.control.tds import TransverseDeflector, StreakingPlane
from esme.control.screens import Screen
from esme.control.sbunches import SpecialBunchesControl
from esme.gui.widgets.common import get_machine_manager_factory, raise_message_box
from esme.maths import line

TDS_FREQUENCY = 3e9

def start_quick_calibrator() -> None:
    app_name = "QCal"
    # this somehow causes big problems...
    # sys.excepthook = make_exception_hook(app_name)
    app = QApplication(sys.argv)
    app.setOrganizationName("lps-tools")
    app.setApplicationName(app_name)

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    main_window = QuickCalibratorWindow()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


# What do I need to calibrate the TDS
# The TDS
# The screen instance
@dataclass
class PhaseScanRanges:
    first: tuple[float | None, float | None] = (None, None)
    second: tuple[float | None, float | None] = (None, None)

    def first_is_defined(self) -> bool:
        return None not in self.first

    def second_is_defined(self) -> bool:
        return None not in self.second


@dataclass
class CrossingResult:
    phases: list[float]
    gauss_pos: list[float] = field(default_factory=list)
    gauss_pos_err: list[float] = field(default_factory=list)
    rms_pos: list[float] = field(default_factory=list)
    rms_pos_err: list[float] = field(default_factory=list)

    def relative_phases(self):
        return self.phases - np.mean(self.phases)
    
    def relative_times(self):
        return self.relative_phases / (360 * TDS_FREQUENCY)

    def fit_gauss(self):
        x = self.relative_times
        y = self.gauss_pos
        yerr = self.gauss_pos_err
        popt, pcov = curve_fit(line, x, y, sigma=yerr, absolute_sigma=True)

        return popt, pcov

    def fit_rms(self):
        x = self.relative_times
        y = self.rms_pos
        yerr = self.rms_pos_err
        popt, pcov = curve_fit(line, x, y, sigma=yerr, absolute_sigma=True)
        return popt, pcov

    def eval_gauss_fit(self) -> tuple[list[float], list[float]]:
        popt, pcov = self.fit_gauss()
        errors = np.sqrt(np.diag(pcov))
        c, m = popt
        cerr, merr = errors
        c = ufloat(c, cerr)
        m = ufloat(m, merr)
        result = c + m * self.relative_times
        values = [r.n for r in result]
        errors = [r.s for r in result]
        return values, errors        

    def time_calibration_gauss(self) -> tuple[float, float]:
        popt, pcov = self.fit_gauss()
        errors = np.sqrt(np.diag(pcov))
        _, m = popt
        _, merr = errors
        return m, merr

    def time_calibration_rms(self) -> tuple[float, float]:
        popt, pcov = self.fit_rms()
        errors = np.sqrt(np.diag(pcov))
        _, m = popt
        _, merr = errors
        return m, merr


@dataclass
class CalibrationResult:
    first: CrossingResult
    second: CrossingResult | None = None


@dataclass
class TimeCalibration:
    amplitude: float
    time_calibration: float
    zero_crossing: float | None = None


class QuickCalibratorWindow(QMainWindow):
    time_calibration = pyqtSignal(TimeCalibration)
    def __init__(self) -> None:
        super().__init__()
        self.ui = Ui_quickcal_window()
        self.ui.setupUi(self)
        self._init_ui()

        self.phase_scan_ranges = PhaseScanRanges()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self._connect_buttons()

        self.threadpool = QThreadPool()
        self.worker = None

    def _update_ui(self):
        pass

    def _init_tds_stack(self):
        stack = self.ui.tds_stack
        factory = get_machine_manager_factory()
        i1tds = factory._get_deflector(DiagnosticRegion.I1)
        stack.set_new_widget_by_region(DiagnosticRegion.I1, MiniTDS(i1tds))

        b2tds = factory._get_deflector(DiagnosticRegion.B2)
        stack.set_new_widget_by_region(DiagnosticRegion.B2, MiniTDS(b2tds))
        self.ui.tds_stack.set_widget_by_region(DiagnosticRegion.I1)

    def _get_tds_from_stack(self) -> TransverseDeflector:
        return self.ui.tds_stack.get_current_widget().tds

    def set_phase_00(self) -> None:
        tds = self._get_tds_from_stack()
        phase = tds.get_phase_sp()
        self.phase_scan_ranges.first = (phase, self.phase_scan_ranges.first[1])
        self.ui.phase_00_label.setText(f"{phase:.2f}")
        self.ui.set_phase_01.setEnabled(True)

    def set_phase_01(self) -> None:
        tds = self._get_tds_from_stack()
        phase = tds.get_phase_sp()
        self.phase_scan_ranges.first = (self.phase_scan_ranges.first[0], phase)
        self.ui.phase_01_label.setText(f"{phase:.2f}")
        self.ui.set_phase_10.setEnabled(True)

    def set_phase_10(self) -> None:
        tds = self._get_tds_from_stack()
        phase = tds.get_phase_sp()
        self.phase_scan_ranges.second = (phase, self.phase_scan_ranges.second[1])
        self.ui.phase_10_label.setText(f"{phase:.2f}")
        self.ui.set_phase_11.setEnabled(True)
    
    def set_phase_11(self) -> None:
        tds = self._get_tds_from_stack()
        phase = tds.get_phase_sp()
        self.phase_scan_ranges.second = (self.phase_scan_ranges.second[0], phase)
        self.ui.phase_11_label.setText(f"{phase:.2f}")

    def _init_plot(self):
        fig = self.ui.calib_plot.fig
        calib_ax = fig.add_subplot(111)
        calib_ax.set_xlabel(r"$\Delta t$ / ps")
        calib_ax.set_ylabel("Position / mm")
        self.calib_ax = calib_ax

    def _init_ui(self):
        self._init_plot()
        self._init_tds_stack()

    def _do_the_calibration(self) -> None:
        if not self.phase_scan_ranges.first_is_defined():
            raise_message_box("Missing phase ranges",
                               informative_text="Cannot calibrate TDS as incomplete phase ranges set, set at least the first phase range pair to proceed.",
                               title="Missing Input",
                               icon="information")
        screen_name = self.ui.area_widget.get_selected_screen_name()
        area = region_from_screen_name(screen_name)
        manager = CalibrationManager(Screen(screen_name),
                                     self._get_tds_from_stack(),
                                     SpecialBunchesControl(area))
        self.worker = CalibrationWorker(manager, self.phase_scan_ranges)
        self.threadpool.start(self.worker)
        self.enable_ui(enable=False)

    def _cancel_calibration(self) -> None:
        self.worker.interrupt_event.set()
        self.enable_ui(enable=True)

    def show_result(self, result: CalibrationResult) -> None:
        self._plot_result(result)
        self._add_plot_text_box(result)
        ### XXX: Some button, which to accept?
        # Instantiate the dialog with two values
        self._calibration_dialog = CalibrationDialog(result)
        # Connect the custom signal to the slot
        self._calibration_dialog.calibrationSelected.connect(self._user_selected_calibration)

        # Show the dialog
        self._calibration_dialog.exec_()

    def _user_selected_calibration(self):
        TimeCalibration()
        pass

    def _plot_result(self, result: CalibrationResult) -> None:
        self._plot_crossing_result(result.first, label="First")
        if self.phase_scan_ranges.second_is_defined():
            self._plot_crossing_result(result.second, label="Second")
        self.calib_ax.legend()

    def _add_plot_text_box(self) -> None:
        pass
        # self.calib_ax.text(0.5, 0.05, fr"First: \sigma_{{}}{}±{} µm/ps")

    def _plot_crossing_result(self, crossing_result: CrossingResult, label) -> None:
        x = crossing_result.first.phases
        # Convert phases to time offsets
        x *= 1e12 # to picoseconds

        y = crossing_result.first.gauss_pos
        yerr = crossing_result.first.gauss_pos_err
        popt, pcov = curve_fit(line, x, y, sigma=yerr, absolute_sigma=True)
        l1, = self.calib_ax.plot(x, line(x, *popt))
        self.calib_ax.errorbar(x, y, yerr=yerr, linestyle="x", color=l1.get_color())

    def _connect_buttons(self) -> None:
        self.ui.set_phase_00.clicked.connect(self.set_phase_00)
        self.ui.set_phase_01.clicked.connect(self.set_phase_01)
        self.ui.set_phase_10.clicked.connect(self.set_phase_10)
        self.ui.set_phase_11.clicked.connect(self.set_phase_11)
        self.ui.start_calib_button.clicked.connect(self._do_the_calibration)
        self.ui.cancel_calib_button.clicked.connect(self._cancel_calibration)

    def enable_ui(self, *, enable: bool) -> None:
        self.ui.area_widget.setEnabled(enable)
        self.ui.sbm_control.setEnabled(enable)
        self.ui.start_calib_button.setEnabled(enable)
        self.ui.tds_stack.setEnabled(enable)
        self.ui.first_phase_range_group_box.setEnabled(enable)
        self.ui.second_phase_range_group_box.setEnabled(enable)
        self.ui.cancel_calib_button.setEnabled(not enable)


class CalibrationSignals(QObject):
    final_result = pyqtSignal(CalibrationResult)

class InterruptedCalibrationError(RuntimeError):
    pass

class CalibrationWorker(QRunnable):
    def __init__(self, man: CalibrationManager, phase_scan_ranges: PhaseScanRanges) -> None:
        super().__init__()
        self.man = man
        self.phase_scan_ranges = phase_scan_ranges
        self.signals = CalibrationSignals()
        self.interrupt_event = Event()
        screenmd = self.man.screen.get_screen_metadata()
        if man.tds.plane is StreakingPlane.HORIZONTAL:
            self.pxsize = screenmd.xsize
        elif man.tds.plane is StreakingPlane.VERTICAL:
            self.pxsize = screenmd.ysize

    def _raise_if_interrupted(self) -> None:
        if self.interrupt_event.is_set():
            raise InterruptedCalibrationError()
        
    def _find_gain(self) -> None:
        anal = self.man.screen.analysis
        self.man.turn_beam_onto_screen()
        time.sleep(1)
        anal.activate_gain_control()
        while anal.is_active():
            self._raise_if_interrupted()
            time.sleep(0.5)
    
    def _take_background(self) -> None:
        self.man.take_beam_off_screen()
        anal = self.man.screen.analysis
        anal.set_background_count(10)
        anal.accumulate_background()
        self._sleep_until_inactive()
        anal.set_subtract_background(do_subtract=True)
    
    def _sleep_until_inactive(self) -> None:
        while self.man.screen.analysis.is_active():
            time.sleep(0.2)
            self._raise_if_interrupted()

    def _do_one_phase_scan(self, crossing_result: CrossingResult) -> None:
        anal = self.man.screen.analysis
        print("STARTING A PHASE SCAN")
        px = self.pxsize
        for phase in crossing_result.phases:
            print(f"{phase=}")
            self.man.tds.set_phase(phase)
            time.sleep(1)
            anal.start_sampling()
            self._sleep_until_inactive()

            xgauss, xgausserr = anal.get_gauss_mean_x()
            xrms, xrmserr = anal.get_rms_mean_x()
            crossing_result.gauss_pos.append(xgauss * px)
            crossing_result.gauss_pos_err.append(xgausserr * px)
            crossing_result.rms_pos.append(xrms * px)
            crossing_result.rms_pos_err.append(xrmserr * px)

        print("phase_scan_finished")

    def _do_phase_scans(self) -> None:
        phases0 = np.linspace(*self.phase_scan_ranges.first, num=5)
        first_crossing = CrossingResult(phases0)
        self._do_one_phase_scan(first_crossing)
        result = CalibrationResult(first_crossing)

        if self.phase_scan_ranges.second_is_defined():
            phases1 = np.linspace(*self.phase_scan_ranges.second, num=5)
            second_crossing = CrossingResult(phases1)
            self._do_one_phase_scan(second_crossing)
            result.second = second_crossing

        self.signals.final_result.emit(result)

    def run(self) -> None:
        # Put beam onto screen if it not already
        anal = self.man.screen.analysis
        self._find_gain()
        self._take_background()
        self.man.turn_beam_onto_screen()
        time.sleep(1)
        self._do_phase_scans()


def _make_cal_string(cal: float, cal_err: float) -> str:
    return f"({cal*1e12:.3g} ± {cal_err*1e12:.3g}) µm/ps"

def show_result_popup(first: tuple[float, float], 
                      second: tuple[float, float] | None = None,
                      parent=None):
    # Create a popup window (QDialog)
    popup = QDialog(parent)
    popup.setWindowTitle("Calibration Finished")
    popup.setFixedSize(300, 150)
    first_str = _make_cal_string(first[0],first[1])
    message = dedent(f"""\
        The Calibration has finished.  Choose a Calibration:

        First: {first_str}
        """)
    if second is not None:
        avg = first[0] + second[0] / 2.0
        avg_err = np.sqrt(first[1] + second[1])
        second_str = _make_cal_string(*second)
        avg_str = _make_cal_string(avg, avg_err)

        message += dedent(f"""\
        Second: {second_str}
        Average: {avg_str}
        """)    

    # Create a label and close button for the popup
    label = QLabel("The calibration has finished.", popup)
    label.setAlignment(Qt.AlignCenter)

    # Add a close button
    button_box = QDialogButtonBox()
    first_button = QPushButton("First")
    second_button = QPushButton("Second")
    average_button = QPushButton("Average")
    cancel_button = QPushButton("Cancel")

    button_box.addButton(first_button, QDialogButtonBox.ActionRole)
    button_box.addButton(second_button, QDialogButtonBox.ActionRole)
    button_box.addButton(average_button, QDialogButtonBox.ActionRole)
    button_box.addButton(cancel_button, QDialogButtonBox.RejectRole)

    # Layout for the popup window
    layout = QVBoxLayout()
    layout.addWidget(label)
    layout.addWidget(button_box)
    popup.setLayout(layout)

    # Show the popup
    popup.exec_()  # This blocks until the popup is closed

        # cal1 = result.first.time_calibration_gauss()
        # if result.second is None:
        #     cal2 = None
        # else:
        #     cal2 = result.second.time_calibration_gauss()


class CalibrationDialog(QDialog):
    # Define a custom signal that will emit the chosen value
    calibrationSelected = pyqtSignal(TimeCalibration)

    # def __init__(self, first: tuple[float, float], second: tuple[float, float] | None = None, parent=None):
    def __init__(self, result: CalibrationResult, parent=None):
        super().__init__(parent)

        # Store the values
        self.result = result
        # self.first = result.first.time_calibration_gauss()
        # self.second = result.second.time_calibration_gauss()

        self._init_ui()

    def _init_ui(self) -> None:
        # Set up the dialog layout
        self.setWindowTitle("Pick a Calibration")
        main_layout = QVBoxLayout()

        cal1 = self.result.first.time_calibration_gauss()
        cal1_str = _make_cal_string(cal1[0], cal1[1])
        message = dedent(f"""\
            The Calibration has finished.  Choose a Calibration:

            First: {cal1_str}
            """)
        
        if self.second is not None:
            avg = self.first[0] + self.second[0] / 2.0
            avg_err = np.sqrt(self.first[1] + self.second[1])
            second_str = _make_cal_string(*self.second)
            avg_str = _make_cal_string(avg, avg_err)

            message += dedent(f"""\
            Second: {second_str}
            Average: {avg_str}
            """)

        # Add a label to the dialog
        message_label = QLabel("The calibration has finished.  Pick a calibration:")
        main_layout.addWidget(message_label)

        # Create the buttons
        self.first_button = QPushButton("First")
        self.second_button = QPushButton("Second")
        self.second_button.setEnabled(self.second is not None)
        self.average_button = QPushButton("Average")
        self.average_button.setEnabled(self.second is not None)
        self.cancel_button = QPushButton("Cancel")

        # Use a QHBoxLayout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.first_button)
        button_layout.addWidget(self.second_button)
        button_layout.addWidget(self.average_button)
        button_layout.addWidget(self.cancel_button)

        # Add the button layout to the main layout
        main_layout.addLayout(button_layout)

        # Set the layout to the dialog
        self.setLayout(main_layout)

        # Disable resizing
        self.setFixedSize(self.sizeHint())

        # Connect buttons to their respective slots
        self.first_button.clicked.connect(self.select_first)
        self.second_button.clicked.connect(self.select_second)
        self.average_button.clicked.connect(self.select_average)
        self.cancel_button.clicked.connect(self.reject)  # Standard QDialog reject behavior for Cancel

    def select_first(self):
        # Emit the 'first' value when First button is clicked
        self.calibrationSelected.emit(self.first)
        self.accept()

    def select_second(self):
        # Emit the 'second' value when Second button is clicked
        self.calibrationSelected.emit(self.second)
        self.accept()

    def select_average(self):
        # Emit the average value when Average button is clicked
        average = (self.first[0] + self.second[0]) / 2
        average_error = np.sqrt(self.first[1]**2 + self.second[1]**2)
        self.calibrationSelected.emit((average, average_error))
        self.accept()