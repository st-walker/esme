from __future__ import annotations

import logging
import pickle
import queue
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QMessageBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage

from esme import DiagnosticRegion
from esme.image import zero_off_axis_regions
from esme.control.exceptions import DOOCSReadError
from esme.control.screens import Position, Screen, ScreenMetadata
from esme.control.tds import StreakingPlane
from esme.gui.ui.imaging import Ui_imaging_widget
from esme.gui.widgets.screen import AxesCalibration
from esme.calibration import AmplitudeVoltageMapping
from scipy.optimize import curve_fit
from esme.gui.widgets.current import CurrentProfilerWindow
from esme.gui.workers import ImagingMessage, ImagePayload, MessageType, ImagingWorker, GaussianParameters

from .common import get_machine_manager_factory, send_widget_to_log

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

pg.setConfigOption("useNumba", True)



class DiagnosticSectionWidget(QWidget):
    def get_section(self) -> DiagnosticRegion:
        if self.i1 is self.mreader:
            return DiagnosticRegion.I1
        elif self.b2 is self.mreader:
            return DiagnosticRegion.B2
        else:
            raise ValueError("Unknown section: %s", self.mreader)

    def get_streaking_plane(self) -> StreakingPlane:
        return self.mreader.deflector.plane


class ImagingControlWidget(DiagnosticSectionWidget):
    """The Image control widget is the widget that gets the image and
    then possibly pushes it to the daughter ScreenWidget.

    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.ui = Ui_imaging_widget()
        self.ui.setupUi(self)

        mfactory = get_machine_manager_factory()
        self.i1, self.b2 = mfactory.make_i1_b2_imaging_managers()
        self.mreader = self.i1

        self.screen = self.mreader.screens["OTRC.58.I1"]
        self._was_beam_on_screen = self.mreader.is_beam_on_screen(self.screen)


        # Thread that reads the images and others can get from.
        self.producer_worker, self.producer_thread = self.setup_data_taking_worker()

        self.current_profiler = CurrentProfilerWindow(self.producer_worker)

        self.elogger = LogBookEntryWriterDialogue(
            self.screen, bg_images=None, parent=self
        )

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.timeout.connect(self._check_screen_position)
        self.timer.start(1000)

        self._connect_buttons()

    def set_tds_calibration(self, voltage_calibration: AmplitudeVoltageMapping):
        # XXX: I should somehow also send this to the elogger so it's clear which
        # Calibration we are using or whatever.
        pass
        # from IPython import embed; embed()
        # self.ui.screen_display_widget.propagate_tds_calibration_signal(
        #     voltage_calibration
        # )

    def _check_screen_position(self) -> None:
        self.producer_worker.submit(
            ImagingMessage(MessageType.CLIP_OFFAXIS,
                              data={"state": self.screen.get_position() is Position.OFFAXIS})
        )

    def _update_ui(self):
        screen = self.screen
        # The button should be active only if the analysis server is not active, i.e.
        # Not doing something already.
        is_active = screen.analysis.is_active()
        self.ui.autogain_button.setEnabled(not is_active)

    def _calculate_dispersion(self) -> None:
        dx, dy = self.mreader.optics.dispersions_at_screen(self.screen.name)
        section = self.get_section()
        if section is DiagnosticRegion.I1:
            self.ui.dispersion_spinner.setValue(dx)
        elif section is DiagnosticRegion.B2:
            self.ui.dispersion_spinner.setValue(dy)
        else:
            raise ValueError("Unknown diagnostic section: %s", section)
        
    def _calculate_time_calibration(self) -> None:
        try:
            time_calibration = self.mreader.calculate_time_calibration(self.screen.name)
        except DOOCSReadError as e:
            QMessageBox.warning(None, "Read Error", f"Error trying to calculate TDS calibration at screen.  Could not read {e.address}.")
            # In case we e.g. cannot read the beam energy (this would typically be if there is no beam...)
            return
        # Multiply by 1e-6 to convert from m/s to µm/ps.
        self.ui.time_calibration_spinbox.setValue(time_calibration * 1e-6)

    def _pass_time_calibration_to_worker(self, time_calibration_value: float) -> None:
        try:
            bunch_charge = self.mreader.optics.get_bunch_charge() * 1e-9 # convert nC to C
        except DOOCSReadError:
            bunch_charge = None
        self.producer_worker.submit(ImagingMessage(MessageType.TIME_CALIBRATION, 
                                                      # Convert µm/ps to m/s.
                                                      data={"time_calibration": time_calibration_value * 1e6,
                                                            "bunch_charge": bunch_charge}))

    def _generate_axes_calibrations(self) -> None:
        dispersion = self.ui.dispersion_spinner.value()
        time_calibration = self.ui.time_calibration_spinbox.value() * 1e6 # convert µm/ps to m/s
        try:
            energy_ev = energy_ev=self.mreader.optics.get_beam_energy() * 1e6
        except DOOCSReadError:
            energy_ev = None
        axescalib = AxesCalibration(
            energy_ev=energy_ev,
            dispersion=dispersion,
            time_calibration=time_calibration,
            streaking_plane=self.mreader.deflector.plane
        )
        self.ui.screen_display_widget.calibrate_axes(axescalib)

    def _connect_buttons(self) -> None:
        self.ui.send_to_logbook_button.clicked.connect(self._open_logbook_writer)
        self.ui.take_background_button.clicked.connect(self.take_background)

        self.ui.autogain_button.clicked.connect(self._activate_auto_gain)
        self.ui.subtract_bg_checkbox.stateChanged.connect(self.set_subtract_background)

        self.ui.play_pause_button.play_signal.connect(self._play_screen)
        self.ui.play_pause_button.pause_signal.connect(self._pause_screen)
        self.ui.read_rate_spinner.valueChanged.connect(self._set_read_frequency)
        self.ui.smooth_image_checkbox.stateChanged.connect(self._set_smooth_image)

        self.ui.calculate_dispersion_button.clicked.connect(self._calculate_dispersion)
        self.ui.calculate_time_calibration_button.clicked.connect(self._calculate_time_calibration)
        self.ui.regenerate_axes_button.clicked.connect(self._generate_axes_calibrations)

        self.ui.time_calibration_spinbox.valueChanged.connect(self._pass_time_calibration_to_worker)

        self.ui.current_profile_button.clicked.connect(self.current_profiler.show)

    def _set_smooth_image(self, smooth_image_state: Qt.CheckState) -> None:
        assert smooth_image_state != Qt.PartiallyChecked
        message = ImagingMessage(MessageType.SMOOTH_IMAGE, 
                                    {"state": bool(smooth_image_state)})
        self.producer_worker.submit(message)

    def _activate_auto_gain(self) -> None:
        # We assume the server is inactive here, as we can only make it active by
        # xxx; technically we should check to see if it is busy here...
        self.screen.analysis.set_clipping(on=self.screen.get_position() is Position.OFFAXIS)

        self.screen.analysis.activate_gain_control()
        self.ui.autogain_button.setEnabled(False)
        # We do not allow the offaxis clipping to be touched as this also touches the image
        # server roi.  the auto gain control is only done in the roi, so if we change the roi
        # whilst doing the autogain on the roi, then we might have a problem.  so just avoid
        # Doing that by disabling this checkbox.
        # We have to clear the background cache when adjusting the gain because we risk
        # subtracting a background at a much higher / lower gain that the beam image, which
        # is basically meaningless.
        message = ImagingMessage(MessageType.CLEAR_BACKGROUND)
        self.producer_worker.submit(message)

    def _open_logbook_writer(self) -> None:
        if tcal := self.ui.time_calibration_spinbox.value():
            self.elogger.time_calibration = tcal * 1e6 # µm/ps to m/s
        else:
            tcal = None
        self.elogger.show_as_new_modal_dialogue(
            self.screen, list(self.producer_worker.bg_images),
            time_calibration=tcal,
            energy_calibration=None
        )

    def set_subtract_background(self, subtract_bg_state: Qt.CheckState) -> None:  # type: ignore
        assert subtract_bg_state != Qt.PartiallyChecked  # type: ignore
        message = ImagingMessage(
            MessageType.SUBTRACT_BACKGROUND, {"state": bool(subtract_bg_state)}
        )
        self.producer_worker.submit(message)

    def take_background(self) -> None:
        message = ImagingMessage(MessageType.CACHE_BACKGROUND, {"number_to_take": 5})
        self._was_beam_on_screen = self.mreader.is_beam_on_screen(self.screen)
        self.mreader.take_beam_off_screen(self.screen)
        self.producer_worker.submit(message)    

    def _play_screen(self) -> None:
        self.producer_worker.submit(ImagingMessage(MessageType.PLAY_SCREEN))

    def _pause_screen(self) -> None:
        self.producer_worker.submit(ImagingMessage(MessageType.PAUSE_SCREEN))

    def _set_read_frequency(self) -> None:
        # TODO: debounce this so only is set after not being touched for a few seconds.
        self.producer_worker.submit(
            ImagingMessage(
                MessageType.SET_FREQUENCY,
                {"frequency": self.ui.read_rate_spinner.value()},
            )
        )

    def setup_data_taking_worker(self) -> tuple[ImagingWorker, QThread]:
        LOG.debug("Initialising screen worker thread")
        producer_worker = ImagingWorker(self.screen)
        # Propagate screen name change to data taking worker
        # responsible for reading from the screen.
        producer_thread = QThread()
        producer_worker.moveToThread(producer_thread)
        producer_worker.display_image_signal.connect(
            self.ui.screen_display_widget.post_beam_image
        )
        producer_worker.nbackground_taken.connect(self._set_nbg_images_acquired_textbox)
        producer_thread.started.connect(producer_worker.run)
        producer_worker.finished_background_acquisition.connect(self._post_background_acquisition)

        producer_thread.start()

        return producer_worker, producer_thread

    def _post_background_acquisition(self) -> None:
        # If the beam was on screen before we started taking background data,
        # Then we putit back on here now.
        if self._was_beam_on_screen:
            self.mreader.turn_beam_onto_screen(self.screen)
            self._was_beam_on_screen = True
        # By default we immediately start subtracking background for convenience.
        self.ui.subtract_bg_checkbox.setCheckState(Qt.Checked)

    def _set_nbg_images_acquired_textbox(self, nbg: int) -> None:
        label = f"Background Images: {nbg}"
        self.ui.nbg_images_acquired_label.setText(label)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.producer_worker.stop_read_loop()
        self.producer_thread.terminate()
        self.producer_thread.wait()

    def set_screen(self, screen_name: str) -> None:
        self.screen = self.mreader.screens[screen_name]
        # Change elogger window screen to the correct screen instance
        self.elogger.set_screen(self.screen)
        # Get the worker producing imagees
        self.producer_worker.change_screen(self.screen)
        self._check_screen_position()
        # Get the daughter Screen widget to also use the relevant screen instance.
        self.ui.screen_display_widget.set_screen(self.screen.name)
        self._calculate_dispersion()
        self._generate_axes_calibrations()



def gauss(x, a, mu, sigma):
         return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

class LogBookEntryWriterDialogue(QDialog):
    DEFAULT_AUTHOR = "xfeloper"

    def __init__(
        self, screen: Screen, bg_images: list[npt.NDArray] | None = None, parent=None
    ) -> None:
        super().__init__(parent=parent)
        self.ui = self._make_ui()

        mfactory = get_machine_manager_factory()
        self.i1reader = mfactory.make_machine_reader_manager(DiagnosticRegion.I1)
        self.b2reader = mfactory.make_machine_reader_manager(DiagnosticRegion.B2)
        # Set initial machine reader instance choice to be for I1.:
        self.mreader = self.i1reader

        # XXX?  could be this be put in QSettings somehow instead?
        hardcoded_path = "/Users/xfeloper/user/stwalker/lps-dumpage/"
        self.outdir = hardcoded_path

        self._screen = screen
        self._time_calibration = None
        self._energy_calibration = None

        self._bg_images = bg_images
        self._executor = ProcessPoolExecutor(max_workers=1)
        # Send a dummy job so that the worker has imported ocelot (slow) etc.
        # and can start taking data straight away if we ask it to later.
        self._executor.submit(lambda: None)

        self._connect_buttons()

    def show_as_new_modal_dialogue(
        self, screen: Screen, bg_images: list[npt.NDArray], 
        time_calibration: float | None = None,
        energy_calibration: float | None = None,
    ) -> None:
        self._screen = screen
        self.bg_images = bg_images
        self._time_calibration = time_calibration
        self._energy_calibration = energy_calibration
        self.ui.nbg_label.setText(f"Background Images: {len(self.bg_images)}")
        self.setModal(True)
        self.show()

    def set_screen(self, screen: Screen):
        self._screen = screen

    def _connect_buttons(self):
        # Connect cancel button to close the window
        self.ui.cancel_button.clicked.connect(self.hide)
        self.ui.start_button.clicked.connect(self.start)

    def _set_ui_enabled(self, enable: bool) -> None:
        self.ui.author_edit.setEnabled(enable)
        self.ui.text_edit.setEnabled(enable)
        self.ui.start_button.setEnabled(enable)
        self.ui.image_spinner.setEnabled(enable)

    def start(self) -> None:
        self._set_ui_enabled(False)
        nimages_to_take = self.ui.image_spinner.value()
        self.ui.progress_bar.setMaximum(nimages_to_take)
        # Get images from data taking thread
        beam_image_futures = self._acquire_n_images_fast_in_subprocess(nimages_to_take)
        self._print_to_logbook_when_done_taking_images(beam_image_futures)

    def _print_to_logbook_when_done_taking_images(
        self, beam_image_futures: list[Future[npt.NDArray]]
    ):
        # We take images at a rate of 10Hz, so if we take 10 images, we expect it to take 1second.
        # so we want our timer to be at somewhere between 10Hz and 1Hz.
        check_period = int(
            (len(beam_image_futures) / 10 / 10) * 1000
        )  # to milliseconds

        def try_and_print_to_logbook():
            # XXX: What if a future raises for some reason?
            # Count number of futures that are finished and update the progress bar
            ndone = sum([future.done() for future in beam_image_futures])
            self.ui.progress_bar.setValue(ndone)

            # Check if we are done, if we aren't, then we try again in $check_period milliseconds
            if ndone != len(beam_image_futures):
                QTimer.singleShot(check_period, try_and_print_to_logbook)
                return

            # OK all our jobs are done, now we can get the results, and print to logbook
            beam_data = [future.result() for future in beam_image_futures]
            self.send_to_xfel_elog(beam_data=beam_data)
            # Turn the UI back on, reset the progress bar and clear the text box...
            self._set_ui_enabled(True)
            self._reset_ui()
            self.hide()

        # Start trying to print to logbook.
        try_and_print_to_logbook()

    def _reset_ui(self):
        # Deliberately do not clear author.
        # And also choose to leave image_spinner value
        # to what it was, as probably repeated use
        # will want the same number of images each time.
        # We also for now deliberately do not clear the log entry either...
        # self.ui.text_edit.clear()
        self.ui.progress_bar.setValue(0)

    def send_to_xfel_elog(self, beam_data: list[dict[str, Any]]) -> None:
        kvps = self.get_machine_state()
        screen_name = kvps["screen"]
        outdir = self._get_output_dir(screen_name)
        outdir.mkdir(parents=True, exist_ok=True)

        # Save background and beam images, if there are any.
        if beam_data:
            with (outdir / "beam_images.pkl").open("wb") as f:
                pickle.dump(beam_data, f)

        if self._bg_images:
            np.savez(outdir / "bg_images.npz", self._bg_images)

        kvps.to_csv(outdir / "channels.csv")

        log_text = self.ui.text_edit.toPlainText()
        screen_info = f"Screen: {screen_name}"
        outdir_info_string = f"Data written to {outdir}"
        "----\n"
        text = "\n----\n".join([log_text, screen_info, outdir_info_string])
        author = self.ui.author_edit.text() or self.DEFAULT_AUTHOR

        # Go all the way to the top of the widget heirarchy for printing.
        parent = self.parent()
        while widget := parent.parent():
            parent = widget
        send_widget_to_log(
            parent, text=text, author=author, severity="MEASURE", title="TDSChum"
        )

    def _get_output_dir(self, screen_name: str) -> Path:
        nowstr = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        location = self._get_location()
        return Path(self.outdir) / location / screen_name / nowstr

    def _get_location(self) -> str:
        return "B2" if self.mreader is self.b2reader else "I1"

    def _acquire_n_images_fast_in_subprocess(self, n: int) -> list[Future[npt.NDArray]]:
        fn = self._screen.get_image_raw_full

        # Submit jobs to the executor.
        futures = []
        for _ in range(n):
            future = self._executor.submit(_get_from_screen_instance, self._screen)
            futures.append(future)

        return futures

    def get_machine_state(self) -> pd.Series:
        screen_name = self._screen.name
        location = self._get_location()
        screen_metadata = self._screen.get_screen_metadata()
        kvps = {"screen": screen_name, 
                "location": location,
                "xpixel_size": screen_metadata.xsize,
                "ypixel_size": screen_metadata.ysize,
                "nxpixels": screen_metadata.nx,
                "nypixels": screen_metadata.ny,
                "time_calibration": self.time_calibration,
                "energy_calibration": self.energy_calibration
                }
        kvps |= self.mreader.full_read()

        # kvps |= self._screen.dump()
        # kvps |= self._screen.analysis.dump()
        series = pd.Series(kvps)
        series.index.name = "channel"
        series.reset_index(name="value")

        return series
        # XXX: TDS CALIBRATION NEEDS TO SOMEHOW BE INCLUDED HERE !!
        # Or elsewhere?

        return MachineState(kvps=series)

    def _make_ui(self) -> SimpleNamespace:
        ui = SimpleNamespace()

        self.setWindowTitle("XFEL e-LogBook Writer")

        # Create the QTextEdit (editable text browser) and set placeholder text
        text_edit = QTextEdit()
        text_edit.setAcceptRichText(False)
        text_edit.setPlaceholderText("Logbook entry...")
        text_edit.setTabChangesFocus(True)
        ui.text_edit = text_edit

        ui.author_label = QLabel("Author")
        ui.author_edit = QLineEdit()
        ui.author_edit.setPlaceholderText("xfeloper")

        # Create the integer spinner with label and progress bar
        ui.images_label = QLabel("Images")
        ui.image_spinner = QSpinBox()
        ui.image_spinner.setValue(10)
        ui.progress_bar = QProgressBar()

        ui.nbg_label = QLabel("Background Images: 0")

        # Create the buttons
        ui.start_button = QPushButton("Start and send to XFEL e-LogBook")
        ui.cancel_button = QPushButton("Cancel")

        # Layouts

        # Author entry
        author_layout = QHBoxLayout()
        author_layout.addWidget(ui.author_label)
        author_layout.addWidget(ui.author_edit)

        # NImages label, nimages spinner, progress bar
        spinner_layout = QHBoxLayout()
        spinner_layout.addWidget(ui.images_label)
        spinner_layout.addWidget(ui.image_spinner)
        spinner_layout.addWidget(ui.progress_bar)

        # Start stop buttons layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(ui.start_button)
        button_layout.addWidget(ui.cancel_button)

        # Putting it all together vertically.
        main_layout = QVBoxLayout()
        main_layout.addWidget(ui.text_edit)
        main_layout.addLayout(author_layout)
        main_layout.addLayout(spinner_layout)
        main_layout.addWidget(ui.nbg_label)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        return ui


def _get_from_screen_instance(screen: Screen) -> npt.NDArray:
    return screen.get_image_raw_full()
