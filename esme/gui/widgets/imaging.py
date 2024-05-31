from __future__ import annotations

import logging
import queue
import tarfile
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import pickle

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QDialog,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QApplication
)

from esme import DiagnosticRegion
from esme.control.machines import MachineReadManager
from esme.control.screens import Screen, ScreenMetadata
from esme.gui.ui.imaging import Ui_imaging_widget
from esme.gui.widgets.screen import ImagePayload
from esme.control.exceptions import DOOCSReadError

from .common import get_machine_manager_factory, send_widget_to_log

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

pg.setConfigOption("useNumba", True)


class MessageType(Enum):
    CACHE_BACKGROUND = auto()
    # TAKE_N_FAST = auto()
    CHANGE_SCREEN = auto()
    SUBTRACT_BACKGROUND = auto()


@dataclass
class DataTakingMessage:
    mtype: MessageType
    data: dict[str, Any] = field(default_factory=dict)


class StopImageAcquisition(Exception):
    pass


@dataclass
class MachineState:
    bg_images: deque[npt.NDArray]
    kvps: pd.Series


class ImagingControlWidget(QWidget):
    """The Image control widget is the widget that gets the image and
    then possibly pushes it to the daughter ScreenWidget.

    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self.ui = Ui_imaging_widget()
        self.ui.setupUi(self)

        mfactory = get_machine_manager_factory()
        self.i1reader: MachineReadManager = mfactory.make_machine_reader_manager(
            DiagnosticRegion.I1
        )
        self.b2reader: MachineReadManager = mfactory.make_machine_reader_manager(
            DiagnosticRegion.B2
        )
        self.mreader = (
            self.i1reader
        )  # Set initial machine reader instance choice to be for I1.

        self.screen_name = "OTRC.55.I1"

        # Thread that reads the images and others can get from.
        self.producer_worker, self.producer_thread = self.setup_data_taking_worker()

        self.elogger = LogBookEntryWriterDialogue(
            self.mreader.screens[self.screen_name], bg_images=None, parent=self
        )

        self._connect_buttons()

    def set_tds_calibration(self, voltage_calibration):
        # XXX: I should somehow also send this to the elogger so it's clear which
        # Calibration we are using or whatever.
        self.ui.screen_display_widget.propagate_tds_calibration_signal(
            voltage_calibration
        )

    def _connect_buttons(self) -> None:
        self.ui.subtract_bg_checkbox.stateChanged.connect(self.set_subtract_background)
        self.ui.send_to_logbook_button.clicked.connect(self._open_logbook_writer)
        self.ui.take_background_button.clicked.connect(self.take_background)

    def _open_logbook_writer(self) -> None:
        screen = self.mreader.screens[self.screen_name]
        self.elogger.show_as_new_modal_dialogue(
            screen, 
            list(self.producer_worker.bg_images)
        )

    def set_subtract_background(self, subtract_bg_state: Qt.CheckState) -> None:  # type: ignore
        assert subtract_bg_state != Qt.PartiallyChecked  # type: ignore
        message = DataTakingMessage(
            MessageType.SUBTRACT_BACKGROUND, {"state": bool(subtract_bg_state)}
        )
        self.producer_worker.submit(message)

    def take_background(self) -> None:
        message = DataTakingMessage(MessageType.CACHE_BACKGROUND, {"number_to_take": 5})
        self.producer_worker.submit(message)

    def setup_data_taking_worker(self) -> tuple[DataTakingWorker, QThread]:
        LOG.debug("Initialising screen worker thread")
        # self.producer_queue: queue.Queue[DataTakingMessage] = queue.Queue()
        # self.consumer_queue: queue.Queue[] = queue.Queue()
        producer_worker = DataTakingWorker(self.mreader.screens[self.screen_name])
        # Propagate screen name change to data taking worker
        # responsible for reading from the screen.
        producer_thread = QThread()
        producer_worker.moveToThread(producer_thread)
        producer_worker.display_image_signal.connect(
            self.ui.screen_display_widget.post_beam_image
        )
        producer_worker.nbackground_taken.connect(self._set_nbg_images_acquired_textbox)
        producer_thread.started.connect(producer_worker.run)

        producer_thread.start()

        return producer_worker, producer_thread

    def _set_nbg_images_acquired_textbox(self, nbg: int) -> None:
        label = f"Background Images: {nbg}"
        self.ui.nbg_images_acquired_label.setText(label)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.producer_worker.stop_read_loop()
        self.producer_thread.terminate()
        self.producer_thread.wait()

    def set_screen(self, screen_name: str) -> None:
        self.screen_name = screen_name
        screen = self.mreader.screens[screen_name]
        # Change elogger window screen to the correct screen instance
        self.elogger.set_screen(screen)
        # Get the worker producing imagees
        self.producer_worker.change_screen(screen)
        # Get the daughter Screen widget to also use the relevant screen instance.
        self.ui.screen_display_widget.set_screen(screen_name)


class DataTakingWorker(QObject):
    display_image_signal = pyqtSignal(ImagePayload)
    nbackground_taken = pyqtSignal(int)

    def __init__(self, initial_screen: Screen):
        super().__init__()
        # The background images cache.
        self.bg_images: deque[npt.NDArray] = deque(maxlen=5)
        self.mean_bg = 0.0

        self.screen: Screen = initial_screen
        self.screen_md: ScreenMetadata = initial_screen.get_screen_metadata()
        self._set_screen(initial_screen)

        # Flag for when possibly emitting a processed image for
        # display: Whether we should subtract the background for the
        # displayed image or not.
        self._do_subtract_background: bool = False
        self._caching_background = False

        # Queue for messages we we receive from
        self._consumerq: queue.Queue[
            tuple[DataTakingMessage]
        ] = queue.Queue()

        self._is_running = False

    def run(self) -> None:
        if self._is_running:
            raise RuntimeError(
                "Already running, run should not be called more than once."
            )
        self._is_running = True
        self._slow_loop()

    def stop_read_loop(self):
        self._consumerq.put(None)

    def _slow_loop(self):
        """This is a slow loop that is the one that is running by default and acquires images at a rate of
        1Hz.
        """
        deadline = time.time() + 0.9
        while True:
            # Only do this event loop about once a second
            wait = deadline - time.time()
            if wait > 0:
                time.sleep(wait)

            # set deadline for next read.  we want to update the screen at a rate of 1Hz.
            # We subtract 0.1s from 1s to get 0.9, because each read takes about 0.1s.
            deadline += 0.9
                            
            # Check for anything in the message queue
            try:
                self._dispatch_from_message_queue()
            except StopImageAcquisition:
                break

            image = self.screen.get_image_raw()
            if image is None:
                # this can happen sometimes, sometimes when switching cameras
                # get_image_raw can for a moment start returning None, of course
                # we need to account for this possibility and go next.
                continue 

            if self._caching_background:
                self.bg_images.appendleft(image)
                nbg_taken = len(self.bg_images)
                self.nbackground_taken.emit(nbg_taken)

                if nbg_taken == self.bg_images.maxlen:
                    # Then we've filled the ring buffer of background images and
                    # can go back to taking beam data, but first cache the mean_bg.
                    self._caching_background = False
                    self.mean_bg = np.mean(self.bg_images, axis=0, dtype=image.dtype)

            self._process_and_emit_image_for_display(image)

    # def _take_n_fast(self, n: int) -> None:
    #     # This completely blocks the thread in that it is no longer listening
    #     # For messages.
    #     out = []
    #     assert n > 0
    #     for i in range(n):
    #         image = self.screen.get_image_raw()
    #         out.append(image)
    #         self.n_fast_state.emit(i)
    #         if (i % 10) == 0:
    #             self._process_and_emit_image_for_display(image)
    #     # At least emit one image for display.
    #     self._process_and_emit_image_for_display(image)
    #     return out

    def submit(self, message: DataTakingMessage) -> None:
        self._consumerq.put(message)

    def _dispatch_from_message_queue(self) -> None:
        # Check for messages from our parent
        # Only messages I account for are switching screen name
        # And switching image taking mode.  We consume all that have been queued
        # before proceeding with image taking.
        while True:
            try:
                message = self._consumerq.get(block=False)
            except queue.Empty:
                return

            if message is None:
                LOG.critical("%s received request to stop image acquisition", type(self).__name__)
                raise StopImageAcquisition("Image Acquisition halt requested.")

            self._handle_message(message)

    def _handle_message(self, message: DataTakingMessage) -> None:
        result = None
        match message.mtype:
            # Could improve this pattern matching here by using both args of DataTakingMessage.
            case MessageType.CACHE_BACKGROUND:
                self._caching_background = True
                self._clear_bg_cache()
            # case MessageType.TAKE_N_FAST:
            #     result = self._take_n_fast(message.data["n"])
            case MessageType.CHANGE_SCREEN:
                self._set_screen(message.data["screen"])
            case MessageType.SUBTRACT_BACKGROUND:
                self._do_subtract_background = message.data["state"]
            case _:
                message = f"Unexpected message send to {type(self).__name__}: {message}"
                LOG.critical(message)
                raise ValueError(message)
        self._consumerq.task_done()

    def _clear_bg_cache(self) -> None:
        self.bg_images = deque(maxlen=self.bg_images.maxlen)
        self.nbackground_taken.emit(0)

    def _process_and_emit_image_for_display(self, image: npt.NDArray) -> None:
        # Subtract background if we have taken some background and enabled it.
        if self.bg_images and self._do_subtract_background:
            image -= self.mean_bg
            image = image.clip(min=0, out=image)
        minpix = np.min(image)
        maxpix = np.max(image)
        xproj = image.sum(axis=1, dtype=np.float32)
        # xproj = scipy.signal.savgol_filter(xproj, window_length=20, polyorder=2)
        yproj = image.sum(axis=0, dtype=np.float32)

        imp = ImagePayload(
            image=image,
            screen_md=self.screen_md,
            levels=(minpix, maxpix),
            xproj=xproj,
            yproj=yproj,
        )

        self.display_image_signal.emit(imp)

    def _read_from_screen(self) -> npt.NDArray:
        image = self.screen.get_image_raw()
        if self._caching_background:
            self.bg_images.appendleft(image)
            if len(self.bg_images) == self.bg_images.maxlen:
                # Then we've filled the ring buffer of background images and
                # can go back to taking beam data, but first cache the mean_bg.
                self._caching_background = False
                self.mean_bg = np.mean(self.bg_images, axis=0, dtype=image.dtype)
        return image

    def _set_screen(self, screen: Screen) -> None:
        self.screen = screen
        # XXX: Possible race condition somewhere here if called from other thread?
        self._clear_bg_cache()
        self._try_and_set_screen_metadata()

    def _try_and_set_screen_metadata(self):
        # if the screen is not powered, then getting the metadata will fail.
        # We will have to just try to get it in the future, hoping that at some
        # point some other part of the program will switch the screen on.
        # So we punt this into the long grass, so to speak.
        try:
            self.screen_md = self.screen.get_screen_metadata()
        except DOOCSReadError:
            timeout = 500
            LOG.warning("Unable to acquire screen metadata, will try again in %ss", timeout / 1000)
            QTimer.singleShot(timeout, self._try_and_set_screen_metadata)

    def change_screen(self, new_screen: Screen) -> None:
        message = DataTakingMessage(MessageType.CHANGE_SCREEN, {"screen": new_screen})
        self._consumerq.put(message)


class LogBookEntryWriterDialogue(QDialog):
    DEFAULT_AUTHOR = "xfeloper"
    def __init__(
        self,
        screen: Screen,
        bg_images: list[npt.NDArray] | None = None,
        parent = None
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

        self._bg_images = bg_images
        self._executor = ProcessPoolExecutor(max_workers=1)
        # Send a dummy job so that the worker has imported ocelot (slow) etc.
        # and can start taking data straight away if we ask it to later.
        self._executor.submit(lambda: None)

        self._connect_buttons()

    def show_as_new_modal_dialogue(self, screen: Screen, bg_images: list[npt.NDArray]) -> None:
        self._screen = screen
        self.bg_images = bg_images
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
        beam_image_futures = self._acquire_n_images_fast_in_subprocess(
            nimages_to_take
        )
        self._print_to_logbook_when_done_taking_images(beam_image_futures)

    def _print_to_logbook_when_done_taking_images(
        self, beam_image_futures: list[Future[npt.NDArray]]
    ):
        # We take images at a rate of 10Hz, so if we take 10 images, we expect it to take 1second.
        # so we want our timer to be at somewhere between 10Hz and 1Hz.  
        check_period = int((len(beam_image_futures) / 10 / 10) * 1000) # to milliseconds

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
        send_widget_to_log(parent, text=text, author=author, severity="MEASURE", title="TDSChum")

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
        kvps = {"screen": screen_name, "location": location}
        kvps |= self.mreader.full_read()
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
