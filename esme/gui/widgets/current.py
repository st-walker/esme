from dataclasses import dataclass, field
from types import SimpleNamespace

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt, QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt5.QtWidgets import QApplication, QVBoxLayout, QWidget, QMainWindow, QFileDialog, QMessageBox
from scipy.constants import c
import time
import numpy as np
import numpy.typing as npt
from threading import Event
from pathlib import Path
from esme.gui.ui.current import Ui_CurrentProfilerWindow
import sys
from esme.plot import _format_df_for_printing
from esme.gui.calibrator import MeasurementContext
from esme.gui.widgets.common import get_machine_manager_factory, get_tds_calibration_config_dir, set_machine_by_region
from esme.control.screens import Screen, Position
from esme.control.pattern import get_bunch_pattern, get_beam_regions

class InterruptedMeasurement(RuntimeError):
    pass




class CurrentProfileWorkerSignals(QObject):
    images_taken_signal = pyqtSignal(int)
    measurement_finished_signal = pyqtSignal()


class CurrentProfileWorker(QRunnable):
    def __init__(self, machine_interface, screen):
        super().__init__()

        self.machine = machine_interface
        self.screen = screen

        self.signals = CurrentProfileWorkerSignals()

        beam_images_per_streak = 10
        bg_images_per_gain = 5

        self.interrupt_event = Event()

        self._images_taken = 0


    def run(self) -> None:
        try:
            self._measure_current_profile()
        except InterruptedMeasurement:
            pass

        self.signals.measurement_finished_signal.emit()

    def n_images_per_measurement(self) -> int:
        return int(self.beam_images_per_streak * 3 + self.bg_images_per_gain * 2)

    def run_screen_autogain(self) -> None:
        self.screen.analysis.activate_gain_control()
        while self.screen.analysis.is_active():
            time.sleep(0.2)

    def _take_images(self, n: int) -> npt.NDArray:
        result = []
        for _ in range(n):
            result.append(self.screen.get_image_raw())
            self._images_taken += 1
            self.signals.images_taken_signal.emit(self._images_taken)
    
    def _measure_current_profile(self) -> None:
        # First turn beam on and do auto gain
        # Then take images.
        # Then turn beam off and take background.
        # Then turn beam on and turn tds on do gain control.
        # Then flip tds phase and take images.

        # Save streak amplitude
        streak_amplitude = self.get_amplitude()

        # Go unstreaked.
        self.machine.deflector.set_amplitude(0.)
        self.machine.turn_beam_onto_screen(self.screen)
        time.sleep(1)
        self.run_screen_autogain()

        unstreaked_beam_images = self._take_images(self.beam_images_per_streak)

        self.machine.take_beam_off_screen(self.screen)
        time.sleep(1)
        bg_unstreaked = self._take_images(self.bg_images_per_gain)

        # Go back to amplitude we are streaking at.
        self.machine.deflector.set_amplitude(streak_amplitude)
        self.run_screen_autogain()

        images_streak0 = self._take_images(n=self.beam_images_per_streak)
        phase0 = self.machine.deflector.get_phase()
        phase1 = phase0 + 180
        self.machine.set_phase(phase1)
        time.sleep(1)
        images_streak1 = self._take_images(n=self.beam_images_per_streak)
        # Go back to original phase
        self.machine.set_phase(phase0)

        self.take_beam_off_screen(self.screen)
        time.sleep(1)

        bg_streaked = self._take_images(n=self.bg_images_per_gain)

@dataclass
class StreakedBeamData:
    phase: float = np.nan
    images: npt.NDArray = None

@dataclass
class UnstreakedBeamData:
    images: npt.NDArray = None
    bg: npt.NDArray = 0.0


@dataclass
class CurrentProfileMeasurement:
    context: MeasurementContext
    bunch_charge: float = np.nan

    streak0: StreakedBeamData = field(default_factory=lambda: StreakedBeamData())
    streak1: StreakedBeamData = field(default_factory=lambda: StreakedBeamData())
    unstreaked: UnstreakedBeamData = field(default_factory=lambda: UnstreakedBeamData())

    background_streaked: npt.ArrayLike = 0.0

    tds_calibration: tuple[list[float], list[float]] | None = None
    tds_calibration_directory: Path | None = None

    def current0(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def current1(self) -> tuple[np.ndarray, np.ndarray]:
        pass


class CurrentProfilerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ui = Ui_CurrentProfilerWindow()
        self.ui.setupUi(self)
        # self.ui.results_table_view.setModel(BeamCurrentTableModel())

        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine

        self.plots = SimpleNamespace()
        self._init_plots()

        self._measurement = None

        self.threadpool = QThreadPool()
        self.worker = None


    def _connect_buttons(self) -> None:
        self.ui.start_measurement_button.clicked.connect(self.measure_current_profile)
        self.ui.cancel_button.clicked.connect(self.cancel_current_profile_measurement)

    def measure_current_profile(self) -> None:
        self.worker = CurrentProfileWorker(self.machine, self._get_screen(), self._active_calibration())
        self.worker.signals.log_message.connect(self._write_to_log)
        self.worker.signals.save_calibration.connect(self.save_calibration)
        self.worker.signals.finished_calibration_setpoint.connect(self._post_calibration_setpoint)
        self.worker.signals.finished_calibration.connect(lambda: self.set_ui_enabled(enabled=True))
        self.worker.signals.calibration_error.connect(self._handle_errors)
        self.threadpool.start(self.worker)
        self.set_ui_enabled(enabled=False)

    def cancel_current_profile_measurement(self) -> None:
        if self.worker is None:
            return
        self.interrupt_event.set()
        self.i1machine.sbunches.stop_diagnostic_bunch()
        self.b2machine.sbunches.stop_diagnostic_bunch()
        # And also turn the beam off if we are on-axis.
        if self._get_screen().get_position() is Position.ONAXIS:
            self.machine.turn_beam_off()
        self.set_ui_enabled(enabled=True)

    def plot(self) -> None:
        pass

    def plot_current_profile(self, current_profile: np.ndarray) -> None:
        pass

    def _connect_buttons(self) -> None:
        self.ui.beamregion_spinner.valueChanged.connect(
            lambda n: self.machine.sbunches.set_beam_region(n - 1)
        )
        self.ui.bunch_number_spinner.valueChanged.connect(
            self.machine.sbunches.set_bunch_number
        )
        self.ui.goto_last_in_beamregion.clicked.connect(self.machine)

    def goto_last_bunch_in_beam_region(self, selected_beam_region: int) -> None:
        beam_regions = get_beam_regions(get_bunch_pattern())
        # This is zero counting!! beam region 1 is 0 when read from sbunch midlayer!
        selected_beam_region = self.machine.sbunches.get_beam_region()
        assert selected_beam_region >= 0
        try:
            br = beam_regions[selected_beam_region]
        except IndexError:
            LOG.info(
                f"User tried to select last bunch of nonexistent beam region: {selected_beam_region}."
            )
            box = QMessageBox(self)  # , "Invalid Beam Region",
            box.setText(f"Beam Region {selected_beam_region+1} does not exist.")
            box.exec()
            return
        else:
            self.dbunch_manager.sbunches.set_bunch_number(br.nbunches())
        self._update_beam_region_and_bunch_ui()


    def clear_displays(self) -> None:
        self.plots.current.clear()
        self.plots.spot_size.clear()

    def _init_plots(self) -> None:
        self.plots.current = self.ui.current_graphics.addPlot(title="Current Profile")
        self.plots.spot_size = self.ui.tilt_graphics.addPlot(
            title="Streaked Spot Sizes"
        )

        self.plots.current.addLegend()
        self.plots.spot_size.addLegend()

        self.plots.current.setLabel("left", "<i>I</i>", units="A")
        self.plots.current.setLabel("bottom", "<i>t</i>", units="s")

        self.plots.spot_size.setLabel("bottom", "<i>S / |S|</i>")
        self.plots.spot_size.setLabel(
            "left", "Spot Size <i>σ<sub>x</sub></i>", units="m"
        )

    def save_result(self) -> None:
        pass

def start_current_profiler(argv) -> None:
    app = QApplication(argv)

    calwindow = CurrentProfilerWindow()
    calwindow.setWindowTitle("Beam Current Profiler")
    calwindow.show()
    calwindow.raise_()
    sys.exit(app.exec_())
