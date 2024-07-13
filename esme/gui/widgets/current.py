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
from esme.calibration import get_tds_com_slope, AmplitudeVoltageMapping
from esme.gui.calibrator import save_measurement_context
from esme.analysis import streaking_parameter, apparent_gaussian_bunch_length_from_processed_image, apparent_rms_bunch_length_from_processed_image
import logging
from esme.calibration import AmplitudeVoltageMapping
from esme.core import DiagnosticRegion
from esme.control.exceptions import DOOCSReadError
from esme.core import DiagnosticRegion, region_from_screen_name

LOG = logging.getLogger(__name__)

import datetime

class InterruptedMeasurement(RuntimeError):
    pass



@dataclass
class StreakedBeamData:
    phase: float = np.nan
    images: npt.NDArray = None
    bg: npt.NDArray = None


@dataclass
class UnstreakedBeamData:
    images: npt.NDArray = None
    bg: npt.NDArray = 0.0


# @dataclass
# class CurrentProfileMeasurement:
#     context: MeasurementContext
#     avmapping: AmplitudeVoltageMapping



@dataclass
class CurrentProfileContext(MeasurementContext):
    avmapping: AmplitudeVoltageMapping = None
    bunch_charge: float = np.nan
    tds_amplitude: float = np.nan

@dataclass
class CurrentProfileMeasurement:
    context: CurrentProfileContext = field(default_factory=lambda: CurrentProfileContext())
    streak0: StreakedBeamData = field(default_factory=lambda: StreakedBeamData())
    streak1: StreakedBeamData = field(default_factory=lambda: StreakedBeamData())
    unstreaked: UnstreakedBeamData = field(default_factory=lambda: UnstreakedBeamData())
    background_streaked: npt.ArrayLike = 0.0

    tds_calibration_directory: Path | None = None

    def current0(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def current1(self) -> tuple[np.ndarray, np.ndarray]:
        pass


class CurrentProfileWorkerSignals(QObject):
    images_taken_signal = pyqtSignal(int)
    measurement_error = pyqtSignal()
    measurement_interrupted_signal = pyqtSignal()
    measurement_finished_signal = pyqtSignal(CurrentProfileMeasurement)

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
            result = self._measure_current_profile()
        except Exception as e:
            self.signals.measurement_error.emit(InterruptedMeasurement())
        else:
            self.signals.measurement_finished_signal.emit(result)

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
        from IPython import embed; embed()
        # Save streak amplitude
        streak_amplitude = self.get_amplitude()

        # Go unstreaked.
        self.machine.deflector.set_amplitude(0.)
        self.machine.turn_beam_onto_screen(self.screen, streak=True)
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
        self.signals.streaked_beam_signal.emit()
        phase1 = phase0 + 180
        self.machine.set_phase(phase1)
        time.sleep(1)
        images_streak1 = self._take_images(n=self.beam_images_per_streak)
        # Go back to original phase
        self.machine.set_phase(phase0)

        self.take_beam_off_screen(self.screen)
        time.sleep(1)

        bg_streaked = self._take_images(n=self.bg_images_per_gain)

        return CurrentProfileMeasurement(streak0=StreakedBeamData(phase0, images_streak0),
                                         streak1=StreakedBeamData(phase1, images_streak1),
                                         unstreaked=UnstreakedBeamData(unstreaked_beam_images,
                                                                      bg_streaked),
                                        background_streaked=bg_streaked)


class CurrentProfilerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ui = Ui_CurrentProfilerWindow()
        self.ui.setupUi(self)

        self.i1machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.I1)
        self.b2machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.B2)
        self.machine = self.i1machine

        self.i1profile = CurrentProfileMeasurement()
        self.b2profile = CurrentProfileMeasurement()

        self.plots = SimpleNamespace()
        self._init_plots()
        self._connect_buttons()

        self._measurement = None

        self.ui.area_control.screen_name_signal.connect(self._set_screen)
        # Set initial screen name in the calibration context.
        self._set_screen(self.ui.area_control.get_selected_screen_name())


        self.threadpool = QThreadPool()
        self.worker = None

    def _update_calibration_context(self) -> None:
        optics_df = self.machine.optics.optics_snapshot()
        screen = self._get_screen()
        screen_name = screen.name
        measurement = self._active_measurement()
        try:
            beam_energy = self.machine.optics.get_beam_energy()
        except DOOCSReadError: # XXX: THIS IS NAUGHTY!!
            beam_energy = 130

        measurement.context.screen_name = screen_name
        measurement.context.snapshot = optics_df
        measurement.context.beam_energy = beam_energy
        # XXX: hardcoded, should be in tds.py class??
        measurement.context.frequency = 3e9
        measurement.context.streaking_plane = self.machine.tds.plane
        try:
            xsize = screen.get_pixel_xsize()
            ysize = screen.get_pixel_ysize()
        except DOOCSReadError:
            xsize = ysize = np.nan

        measurement.context.pixel_sizes = xsize, ysize
        measurement.context.screen_position = screen.get_position()

        xminmax = screen.analysis.get_xroi_clipping()
        yminmax = screen.analysis.get_yroi_clipping()
        measurement.context.off_axis_roi_bounds = (xminmax, yminmax)

    def _set_region(self, region: DiagnosticRegion) -> None:
        if region is DiagnosticRegion.I1:
            self.machine = self.i1machine
        elif region is DiagnosticRegion.B2:
            self.machine = self.b2machine

    def _set_screen(self, screen_name: str) -> None:
        # there shoudl be some logic here, if a b2 screen, set screen for b2 calibration
        region = region_from_screen_name(screen_name)
        # If an i1 screen, set the screen for the i1 calibration instance, or if a b2
        # calibration then do it for the b2 instance.
        if region is DiagnosticRegion.I1:
            self.i1profile.context.screen_name = screen_name
        elif region is DiagnosticRegion.B2:
            self.b2profile.context.screen_name = screen_name
        # Set machine instance to be used.
        self._update_calibration_context()
        self._set_region(region)
        self._update_calibration_parameters_ui()

    def _update_calibration_parameters_ui(self) -> None: 
        ctx = self._active_measurement().context
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

    def _active_measurement(self) -> CurrentProfileMeasurement:
        if self.i1machine is self.machine:
            return self.i1profile
        elif self.b2machine is self.achime:
            return self.b2profile
        raise RuntimeError()

    def _connect_buttons(self) -> None:
        self.ui.start_measurement_button.clicked.connect(self.measure_current_profile)
        self.ui.cancel_button.clicked.connect(self.cancel_current_profile_measurement)
        self.ui.beam_region_spinner.valueChanged.connect(
            lambda n: self.machine.sbunches.set_beam_region(n - 1)
        )
        self.ui.bunch_number_spinner.valueChanged.connect(
            self.machine.sbunches.set_bunch_number
        )
        self.ui.goto_last_in_beamregion.clicked.connect(lambda: self.sbunches.set_to_last_bunch())

    def _show_measurement_parameters(self) -> None:
        model = self.ui.beam_table_view.model
        gauss = model.gaussian_params
        rms = model.rms_params

        measurement = self._active_measurement()
        ctx = measurement.context

        r12_streaking = ctx.r12_streaking

        mean_streaked_background = np.mean(measurement.background_streaked, axis=1)
        
        for image in measurement.streak0.images:
            pass
        #XXX: THIS ALL NEEDS FIXING / DOING.
        # apparent_rms_bunch_length_from_processed_image()

    def _get_screen(self) -> Screen:
        return self.machine.screens[self.i1profile.context.screen_name]

    def measure_current_profile(self) -> None:
        self.worker = CurrentProfileWorker(self.machine, self._get_screen())
        self.clear_displays()
        # Progress bar stuff:
        self.ui.progress_bar.setValue(0)
        self.ui.progress_bar.setMaximum(self.worker.n_images_per_measurement())
        self.worker.signals.images_taken_signal.connect(self.ui.progress_bar.setValue)
        # Enabling the UI again at when the measurement stops.
        self.worker.signals.measurement_finished_signal.connect(lambda: self.set_ui_enabled(enabled=True))

        self.worker.signals.streaked_beam_signal.connect(self._post_streaked_beam)
        self.worker.signals.unstreaked_beam_data.connect(self._post_unstreaked_beam)
        self.worker.signals.measurement_finished_signal.connect(self._post_final_resut)
        self.worker.run()
        # self.threadpool.start(self.worker)
        self.set_ui_enabled(enabled=False)

    def _post_unstreaked_beam(self, data: UnstreakedBeamData) -> None:
        this_measurement = self._active_measurement()
        this_measurement.unstreaked = data

    def _post_streaked_beam(self, data: StreakedBeamData) -> None:
        this_measurement = self._active_measurement()
        # Set data.
        if this_measurement.streak0.images is None:
            this_measurement.streak0 = data
        elif this_measurement.streak1.images is None:
            this_measurement.streak1 = data

        ctx = this_measurement.context
        current_profile = self.sum(axis=1) # XXX: Check this!

        pxsize = ctx.get_streaked_pixel_size()

        ampl_v_mapping = self.self._active_voltage_mapping()
        voltage = this_measurement.avmapping.get_voltage(ctx.tds_amplitude)
        r12_streaking = ctx.r12_streaking

        # Convert pixels to time axis:
        calibration_factor = get_tds_com_slope(r12_streaking, 
                                               ctx.beam_energy,
                                               voltage)
        time = np.arange(len(current_profile)) * pxsize
        time-= time[-1] / 2.0
        time /= calibration_factor

        self.plots.current.plot(time, current_profile)

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
        outdir = self._get_outdir()
        self._save_context(outdir)
        self._save_images(outdir)

    def _get_outdir(self) -> Path:
        # Something sensible should go here...
        nowstr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        region = self._get_active_region().name.lower()
        # XXX: change this hardcoded path to my own space...
        basedir = Path("/Users/xfeloper/user/stwalker/lps-measurements")
        return basedir / region / nowstr
    
    def _save_images(self, outdir: Path) -> None:
        measurement = self._active_measurement()
        first_streak = measurement.streak0
        second_streak = measurement.streak1
        measurement.background_streaked

        # Save background for streaked images
        np.save_compressed(outdir / "background_streaked.npz",
                           images=measurement.background.streaked)
        # Save unstreaked images with unstreaked background
        np.save_compressed(outdir / "unstreaked.npz",
                           images=measurement.unstreaked.images,
                           bg=measurement.unstreaked.bg)
        # Save first streak with phase
        np.savez_compressed(outdir / "streak0.npz",
                            phase=measurement.streak0.images,
                            images=measurement.streak0.phase)
        # Save other streak with phase.
        np.savez_compressed(outdir / "streak1.npz",
                            phase=measurement.streak1.phase,
                            images=measurement.streak1.images)
        
    def _save_context(self, outdir: Path) -> None:
        measurement = self._active_measurement()
        amplitude = measurement.tds_amplitude
        bunch_charge = measurement.bunch_charge
        ctx = self._active_measurement().context
        avmapping = ctx.avmapping
        amplitudes = avmapping._amplitudes
        voltages = avmapping._voltages
        calibration_directory = str(measurement.tds_calibration_directory)
        save_measurement_context(ctx.context,
                                 tds_amplitude=amplitude,
                                 bunch_charge=bunch_charge,
                                 calibration_amplitudes=amplitudes,
                                 camplibration_voltages=voltages,
                                 calibration_path=calibration_directory)
        
#tds_calibration: tuple[list[float], list[float]] | None = None
#    tds_calibration_directory


def start_current_profiler(argv) -> None:
    app = QApplication(argv)

    calwindow = CurrentProfilerWindow()
    calwindow.setWindowTitle("Beam Current Profiler")
    calwindow.show()
    calwindow.raise_()
    sys.exit(app.exec_())
