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
from esme.gui.calibrator import save_measurement_context
from esme.analysis import streaking_parameter, apparent_gaussian_bunch_length_from_processed_image, apparent_rms_bunch_length_from_processed_image
import logging
from esme.core import DiagnosticRegion
from esme.control.exceptions import DOOCSReadError
from esme.core import DiagnosticRegion, region_from_screen_name
from esme.gui.workers import ImagingWorker, ImagingMessage, ImagePayload, MessageType
import queue
from collections import deque

LOG = logging.getLogger(__name__)

import datetime

class InterruptedMeasurement(RuntimeError):
    pass

class MeasurementError(RuntimeError):
    def __init__(self, reason, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reason = reason


@dataclass
class StreakedBeamData:
    phase: float = np.nan
    image_payloads: list[ImagePayload] = None

    def mean_current_profile(self):
        return np.mean([pl.current for pl in self.image_payloads], axis=0) # XXX: is this correct?
    
    def mean_stdev_bunch_length(self) -> float:
        return np.mean([pl.stdev_bunch_length for pl in self.image_payloads])

    def mean_gaussian_bunch_length(self) -> float:
        gauss_sigmas = []
        for pl in self.image_payloads:
            param = pl.gaussfit_current
            gauss_sigmas.append(param.sigma)
        return np.mean(gauss_sigmas)

    def time_calibrated(self):
        return self.image_payloads[0].time_calibrated
    
    def time(self)
        return self.image_payloads[0].time
        
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
    unstreaked: list[ImagePayload] = field(default_factory=list)
    background_unstreaked: npt.ArrayLike | None = None
    background_streaked: npt.ArrayLike | None = None

    tds_calibration_directory: Path | None = None

    def current0(self) -> tuple[np.ndarray, np.ndarray]:
        return self.streak0.time, self.streak0.mean_current_profile()

    def current1(self) -> tuple[np.ndarray, np.ndarray]:
        return self.streak1.time, self.streak1.mean_current_profile()
    
    def unstreaked_stdev_beam_length(self) -> float:
        return np.mean([pl.stdev_bunch_length for pl in self.unstreaked])
    
    def unstreaked_gauss_beam_length(self) -> float:
        sizes = []
        for pl in self.unstreaked:
            param = pl.gaussfit_current
            sizes.append(param.sigma)
        return np.mean(sizes)


class CurrentProfileWorkerSignals(QObject):
    images_taken_signal = pyqtSignal(int)
    measurement_error = pyqtSignal(MeasurementError)
    measurement_interrupted_signal = pyqtSignal()
    measurement_finished_signal = pyqtSignal(CurrentProfileMeasurement)

class CurrentProfileWorker(QRunnable):
    def __init__(self, machine_interface, screen, imager_worker):
        super().__init__()

        self.machine = machine_interface
        self.screen = screen
        self.imager_worker = imager_worker

        self.signals = CurrentProfileWorkerSignals()

        self.beam_images_per_streak = 10
        self.bg_images_per_gain = self.imager_worker.bg_images.maxlen

        self.interrupt_event = Event()

        self._images_taken = 0

    def run(self) -> None:
        try:
            result = self._measure_current_profile()
        except InterruptedMeasurement:
            self.signals.measurement_interrupted_signal.emit()
        except Exception as e
            self.signals.measurement_error.emit(InterruptedMeasurement(repr(e)))
        else:
            self.signals.measurement_finished_signal.emit(result)

    def _raise_if_interrupted(self, read_rate) -> None:
        if self.interrupt_event.is_set():
            # restore read rate and raise exception to unwind stack and cancel measurement.
            self.imager_worker.submit(ImagingMessage(MessageType.SET_FREQUENCY, data={"frequency": read_rate}))
            raise InterruptedMeasurement()

    def n_images_per_measurement(self) -> int:
        return int(self.beam_images_per_streak * 3 + self.bg_images_per_gain * 2)

    def run_screen_autogain(self) -> None:
        self.screen.analysis.activate_gain_control()
        while self.screen.analysis.is_active():
            time.sleep(0.2)

    def _take_images(self, n: int) -> list[ImagePayload]:
        initial_read_rate = self.imager_worker.read_frequency
        LOG.debug("Setting image analysis thread to max rate (10Hz) for data acquisition")
        self.imager_worker.submit(ImagingMessage(MessageType.SET_FREQUENCY, data={"frequency": 10.0}))
        # Subscribe to the image worker and get a queue.
        image_queue = self.imager_worker.subscribe(n)
        images = []
        while image_queue.num_images_remaining != 0:
            self._raise_if_interrupted(initial_read_rate)
            try:
                images.append(image_queue.q.get(timeout=1))
                image_queue.task_done()
            except queue.Empty:
                # We were too fast, sleep and let some more images get taken.
                time.sleep(0.2)
        self.signals.images_taken_signal.emit(n)
        # We've finished, now we go back to the previous read rate.
        LOG.debug("Post image acquisition; reverting to initial read rate: %s", initial_read_rate)
        self.imager_worker.submit(ImagingMessage(MessageType.SET_FREQUENCY, data={"frequency": initial_read_rate}))
        return images
    
    def _take_background(self) -> deque[npt.NDArray]:
        # _take_background is different to _take_images in that we simply
        # clear the background cache, go to max rate, read the cache,
        # And then revert the read frequency back to what it was before.

        # Save read frequency for future reference
        initial_read_rate = self.imager_worker.read_frequency
        # Go to fast read rate.
        self.imager_worker.submit(ImagingMessage(MessageType.SET_FREQUENCY, data={"frequency": 10.0}))
        # Clear the background cache.
        self.imager_worker.submit(ImagingMessage(MessageType.CLEAR_BACKGROUND))
        # Wait for background clearing to be done.
        while not self.imager_worker.bg_cache_is_empty():
            self._raise_if_interrupted()
            time.sleep(0.1)
        LOG.info("Cleared background cache, starting to take background in acquisition thread...")
        # Start caching background again.
        self.imager_worker.submit(ImagingMessage(MessageType.CACHE_BACKGROUND))
        while not self.imager_worker.bg_cache_is_full():
            self._raise_if_interrupted(initial_read_rate)
            time.sleep(0.1)
        LOG.info("Finished taking background.  Reverting to initial read rate: %s", initial_read_rate)
        # Go back to the original read rate.
        self.imager_worker.submit(ImagingMessage(MessageType.SET_FREQUENCY, data={"frequency": initial_read_rate}))
        # Get the cached background and return it.
        bg_cache = self.imager_worker.get_cached_background()
        self.signals.images_taken_signal.emit(len(bg_cache))
        return bg_cache
    
    def _measure_current_profile(self) -> None:
        self.producer_worker.submit(message)
        # First turn beam on and do auto gain
        # Then take images.
        # Then turn beam off and take background.
        # Then turn beam on and turn tds on do gain control.
        # Then flip tds phase and take images.

        # Save streak amplitude
        streak_amplitude = self.machine.tds.get_amplitude_rb()
        LOG.info("Setting amplitude to zero to measure unstreaked beam size")
        # Go unstreaked.
        self.machine.tds.set_amplitude(0.)
        self.machine.turn_beam_onto_screen(self.screen, streak=True)
        time.sleep(1)
        self.run_screen_autogain()
        LOG.info("Taking %s unstreaked beam images...", self.beam_images_per_streak)
        unstreaked_beam_images = self._take_images(n=self.beam_images_per_streak)

        self.machine.take_beam_off_screen(self.screen)
        time.sleep(1)
        bg_unstreaked = self._take_background()
        # Go back to amplitude we are streaking at.
        self.machine.tds.set_amplitude(streak_amplitude)
        self.run_screen_autogain()

        print("Finished with auogain round 2")

        streaked_payloads0 = self._take_images(n=self.beam_images_per_streak)
        print("finished first unstreaked batch!")
        phase0 = self.machine.tds.get_phase()
        self.signals.streaked_beam_signal.emit()
        phase1 = phase0 + 180
        self.machine.set_phase(phase1)
        time.sleep(1.5)
        streaked_payloads1 = self._take_images(n=self.beam_images_per_streak)
        # Go back to original phase
        self.machine.set_phase(phase0)
        self.take_beam_off_screen(self.screen)
        time.sleep(1)

        bg_streaked = self._take_background()

        return CurrentProfileMeasurement(streak0=StreakedBeamData(phase0, streaked_payloads0),
                                         streak1=StreakedBeamData(phase1, streaked_payloads1),
                                         unstreaked=UnstreakedBeamData(unstreaked_beam_images,
                                                                      bg_streaked),
                                        background_streaked=bg_streaked)


class CurrentProfilerWindow(QMainWindow):
    def __init__(self, data_acquisition_worker=None) -> None:
        super().__init__()

        self.ui = Ui_CurrentProfilerWindow()
        self.ui.setupUi(self)

        self.i1machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.I1)
        self.b2machine = get_machine_manager_factory().make_tds_calibration_manager(DiagnosticRegion.B2)
        self.machine = self.i1machine

        self.data_acquisition_worker = data_acquisition_worker

        self.i1profile = CurrentProfileMeasurement()
        self.b2profile = CurrentProfileMeasurement()

        self.plots = SimpleNamespace()
        self._init_plots()
        self._connect_buttons()

        self._measurement = None
        # Set initial screen name in the calibration context.

        self.threadpool = QThreadPool()
        self.worker = None

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
        self._set_region(region)

    def _active_measurement(self) -> CurrentProfileMeasurement:
        if self.i1machine is self.machine:
            return self.i1profile
        elif self.b2machine is self.machine:
            return self.b2profile
        raise RuntimeError()

    def _connect_buttons(self) -> None:
        self.ui.start_measurement_button.clicked.connect(self.measure_current_profile)
        self.ui.cancel_button.clicked.connect(self.cancel_current_profile_measurement)

    def _get_screen(self) -> Screen:
        return self.machine.screens[self.i1profile.context.screen_name]

    def measure_current_profile(self) -> None:
        self.worker = CurrentProfileWorker(self.machine, self._get_screen(), self.data_acquisition_worker)
        self.clear_displays()
        # Progress bar stuff:
        self.ui.progress_bar.setValue(0)
        self.ui.progress_bar.setMaximum(self.worker.n_images_per_measurement())
        self.worker.signals.images_taken_signal.connect(self._increment_progress_bar)

        # Enabling the UI again at when the measurement stops.
        # self.worker.signals.measurement_finished_signal.connect(lambda: self.set_ui_enabled(enabled=True))
        self.worker.signals.measurement_finished_signal.connect(self._post_final_result)
        # self.worker.run() # Use this line instead of the threadpool to run sequentially.
        self.threadpool.start(self.worker)
        self.set_ui_enabled(enabled=False)

    def _increment_progress_bar(self, increment: int) -> None:
        self.ui.progress_bar.setValue(self.ui.progress_bar.value() + increment)

    def set_ui_enabled(self, *, enabled) -> None:
        self.ui.start_measurement_button.setEnabled(enabled)
        self.ui.cancel_button.setEnabled(not enabled)        

    def _post_final_result(self, result: CurrentProfileMeasurement):
        if self.i1machine is self.machine:
            self.i1profile = result
        elif self.b2machine is self.machine:
            self.b2profile = result
        else:
            raise TypeError()

        self._plot_current_profile(result)        
        self._plot_spotsizes(result)
        self._result_to_table(result)

    def _result_to_table(self, result: CurrentProfileMeasurement) -> None:
        self.ui.beam_table_view.gaussian_params.sigma_t = 1
        self.ui.beam_table_view.gaussian_params.resolution_t = 1
        self.ui.beam_table_view.gaussian_params.sigma_x0 = 1
        self.ui.beam_table_view.gaussian_params.sigma_xi = 1
                
        self.ui.beam_table_view.rms_params.sigma_t = 1
        self.ui.beam_table_view.rms_params.resolution_t = 1
        self.ui.beam_table_view.rms_params.sigma_x0 = 1
        self.ui.beam_table_view.rms_params.sigma_xi = 1

    def _plot_current_profile(self, result: CurrentProfileMeasurement) -> None:
        self._plot_streaked_beam_current(result.streak0)
        self._plot_streaked_beam_current(result.streak1)
        self._plot_beam_two_point_analysis(result.streak0, result.streak1)

    def _plot_streaked_beam_current(self, data: StreakedBeamData) -> None:
        currents = [im.current for im in data.images]
        mean_current = np.mean(currents, axis=1) # XXX: is this correct? possibly not.
        self.plots.current.plot(time, mean_current, name=f"𝜙 = {data.phase}°")

    def _plot_beam_two_point_analysis(self, beam0: StreakedBeamData, beam1: StreakedBeamData) -> None:
        # !! XXX: two point analysis goes here..
        pass

    def _plot_spotsizes(self, result: CurrentProfileMeasurement) -> None:
        gauss_size0 = result.streak0.mean_gaussian_bunch_length()
        stdev_size0 = result.streak0.mean_stdev_bunch_length()
        gauss_size1 = result.streak1.mean_gaussian_bunch_length()
        stdev_size1 = result.streak1.mean_stdev_bunch_length()
        gauss_size_unstreaked = result.unstreaked_gauss_beam_length()
        stdev_size_unstreaked = result.unstreaked_stdev_beam_length()

        # XXX: These first two plots should be scatter plots.
        self.plots.spot_size.plot([-1, 0, 1], [gauss_size0,
                                               gauss_size_unstreaked,
                                               gauss_size1],
                                               "Gauss.")
        self.plots.spot_size.plot([-1, 0, 1], [stdev_size0,
                                               stdev_size_unstreaked,
                                               stdev_size1],
                                               name="stdev")
        # Now we do the fitting.
        

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
