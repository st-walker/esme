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
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from esme import DiagnosticRegion
from esme.control.machines import MachineReadManager
from esme.control.screens import Screen, ScreenMetadata
from esme.gui.ui.imaging import Ui_imaging_widget
from esme.gui.widgets.screen import ImagePayload

from .common import get_machine_manager_factory, send_widget_to_log

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

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

        self.elogger_window = LogBookEntryWriterDialogue(
            self.mreader.screens[self.screen_name]
        )

        self.connect_buttons()

    def set_tds_calibration(self, voltage_calibration):
        # XXX: I should somehow also send this to the elogger so it's clear which
        # Calibration we are using or whatever.
        self.ui.screen_display_widget.propagate_tds_calibration_signal(
            voltage_calibration
        )

    def connect_buttons(self) -> None:
        self.ui.subtract_bg_checkbox.stateChanged.connect(self.set_subtract_background)
        self.ui.send_to_logbook_button.clicked.connect(self._open_logbook_writer)
        self.ui.take_background_button.clicked.connect(self.take_background)

    def _open_logbook_writer(self) -> None:
        self.elogger_window.bg_images = list(self.producer_worker.bg_images)
        self.elogger_window.show()

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
        producer_thread.start()

        return producer_worker, producer_thread

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.producer_worker.stop_read_loop()
        self.producer_thread.terminate()
        self.producer_thread.wait()

    def set_screen(self, screen_name: str) -> None:
        self.screen_name = screen_name
        screen = self.mreader.screens[screen_name]
        # Change elogger window screen to the correct screen instance
        self.elogger_window.set_screen(screen)
        # Get the worker producing imagees
        self.producer_worker.change_screen(screen)
        # Get the daughter Screen widget to also use the relevant screen instance.
        self.ui.screen_display_widget.set_screen(screen_name)


class DataTakingWorker(QObject):
    display_image_signal = pyqtSignal(ImagePayload)
    n_fast_state = pyqtSignal(int)

    def __init__(self, initial_screen: Screen):
        super().__init__()
        # The background images cache.
        self.bg_images: deque[npt.NDArray] = deque(maxlen=5)
        self.mean_bg = 0.0

        self.screen: Screen
        self.screen_md: ScreenMetadata
        self._set_screen(initial_screen)

        # Flag for when possibly emitting a processed image for
        # display: Whether we should subtract the background for the
        # displayed image or not.
        self._do_subtract_background: bool = False
        self._caching_background = False

        # Queue for messages we we receive from
        self._consumerq: queue.Queue[
            tuple[DataTakingMessage, Future[Any]]
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
        """This is the slow loop that is the one that is running by default and acquires images at
        1Hz.
        """
        # i = 0
        deadline = time.time()

        while True:
            # Check for anything in the message queue
            try:
                self._dispatch_from_message_queue()
            except StopImageAcquisition:
                break

            if time.time() < deadline:
                continue

            image = self.screen.get_image()
            if self._caching_background:
                self.bg_images.appendleft(image)
                if len(self.bg_images) == self.bg_images.maxlen:
                    # Then we've filled the ring buffer of background images and
                    # can go back to taking beam data, but first cache the mean_bg.
                    self._caching_background = Fase
                    self.mean_bg = np.mean(self.bg_images, axis=0, dtype=image.dtype)

            self._process_and_emit_image_for_display(image)

            deadline = time.time() + 0.9  # add 1-0.1.

    # def _take_n_fast(self, n: int) -> None:
    #     # This completely blocks the thread in that it is no longer listening
    #     # For messages.
    #     out = []
    #     assert n > 0
    #     for i in range(n):
    #         image = self.screen.get_image()
    #         out.append(image)
    #         self.n_fast_state.emit(i)
    #         if (i % 10) == 0:
    #             self._process_and_emit_image_for_display(image)
    #     # At least emit one image for display.
    #     self._process_and_emit_image_for_display(image)
    #     return out

    def submit(self, message: DataTakingMessage) -> Future[Any]:
        future: Future[Any] = Future()
        # Disallow cancellation by the client so immediately set it to running, once
        # a message is in the queue it will be acted on eventually.
        future.set_running_or_notify_cancel()
        self._consumerq.put((message, future))
        return future

    def _dispatch_from_message_queue(self) -> None:
        # Check for messages from our parent
        # Only messages I account for are switching screen name
        # And switching image taking mode.  We consume all that have been queued
        # before proceeding with image taking.
        while True:
            try:
                message, future = self._consumerq.get(block=False)
            except queue.Empty:
                return

            if message is None:
                raise StopImageAcquisition("Image Acquisition halt requested.")

            self._handle_message(message, future)

    def _handle_message(
        self, message: DataTakingMessage, future: Future[None | list[npt.NDArray]]
    ) -> None:
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
                raise ValueError(
                    f"Unexpected message send to Image Producing thread: {message}"
                )
        future.set_result(result)
        self._consumerq.task_done()

    def _clear_bg_cache(self) -> None:
        self.bg_images = deque(maxlen=self.bg_images.maxlen)

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
        image = self.screen.get_image()
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
        self.screen_md = screen.get_screen_metadata()

    def change_screen(self, new_screen: Screen) -> None:
        self._consumerq.put(
            (
                DataTakingMessage(MessageType.CHANGE_SCREEN, {"screen": new_screen}),
                Future(),
            )
        )


class LogBookEntryWriterDialogue(QMainWindow):
    def __init__(
        self, screen: Screen, bg_images: list[npt.NDArray] | None = None
    ) -> None:
        super().__init__()
        self.ui = self._make_ui()

        mfactory = get_machine_manager_factory()
        self.i1reader = mfactory.make_machine_reader_manager(DiagnosticRegion.I1)
        self.b2reader = mfactory.make_machine_reader_manager(DiagnosticRegion.B2)
        self.mreader = (
            self.i1reader
        )  # Set initial machine reader instance choice to be for I1.

        # XXX?  could be this be put in QSettings somehow instead?
        hardcoded_path = "/Users/xfeloper/user/stwalker/lps-dumpage/"
        self.outdir = hardcoded_path

        self._screen = screen

        self._bg_images = bg_images
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._timer = QTimer()

        self.connect_buttons()

    def set_screen(self, screen: Screen):
        self._screen = screen

    def _connect_buttons(self):
        # Connect cancel button to close the window
        self.ui.cancel_button.clicked.connect(self.hide)
        self.ui.start_button.clicked.connect(self.start)

    def start(self) -> None:
        self.ui.progress_bar.setMaximum(self.ui.image_spinner.value())
        # Get images from data taking thread
        beam_image_futures = self._acquire_n_images_fast_in_subprocess(
            self.ui.image_spinner.value()
        )
        self.send_to_xfel_elog(beam_image_futures)
        self._timer.timeout.connect(
            lambda: self._print_to_logbook_when_ready(beam_images)
        )
        self._timer.start(250)

    def _print_to_logbook_when_ready(
        self, beam_image_futures: list[Future[npt.NDArray]]
    ):
        if not all([future.completed() for future in beam_image_futures]):
            return
        self._timer.stop()
        beam_images = [future.result() for future in beam_image_futures]
        self.send_to_xfel_elog(beam_images=beam_images)

    def send_to_xfel_elog(self, beam_images: list[npt.NDArray]) -> None:
        mstate = self.get_machine_state()
        outpath = self._get_output_path(
            mstate.kvps["location"], mstate.kvps["screen_name"]
        )

        # # with tarfile.open(outpath, "w:gz") as tarball:  # With compression
        # if i do compress, don't forget .gz at the end.
        with tarfile.open(f"{outpath}.tar", "w") as tarball:  # No compression for now
            # Pickle is about 5x faster but for now prefer np.savez because I
            # don't know what the long term implications are for using pickle
            # e.g. will some file still be readable in a few year's time?
            buffer = BytesIO()
            np.savez(buffer, beam_images=beam_images, bg_images=self._bg_images)
            buffer.seek(0)
            images_tarinfo = tarfile.TarInfo(name=f"{outpath}/images.npz")
            tarball.addfile(images_tarinfo, fileobj=buffer)
            buffer.seek(0)
            # Now write all the various DOOCS addresses we've read to file
            kvps_tarinfo = tarfile.TarInfo(name=f"{outpath}/channels.csv")
            mstate.kvps.to_csv(buffer, index=False)
            tarball.addfile(kvps_tarinfo, fileobj=buffer)

        text = f"{self.ui.text_edit.toPlainText()}\n\n Data written to {outpath}"
        send_widget_to_log(self.window(), text=text)

    def _get_output_path(self, location: str, screen_name: str) -> Path:
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        return Path(self.outdir) / location / screen_name / f"{time}.tar"

    def _get_location(self) -> str:
        return "B2" if self.mreader is self.b2reader else "I1"

    def _acquire_n_images_fast_in_subprocess(self, n: int) -> list[Future[npt.NDArray]]:
        fn = self._screen.get_image

        def increment_progress_bar(_):
            self.ui.progress_bar.setValue(self.ui.progress_bar.value() + 1)

        # Submit jobs to the executor.
        futures = []
        for _ in range(n):
            future = self._executor.submit(_get_from_screen_instance, self._screen)
            future.add_done_callback(increment_progress_bar)
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
        ui.text_edit = text_edit

        # Create the integer spinner with label and progress bar
        ui.images_label = QLabel("Images")
        ui.image_spinner = QSpinBox()
        ui.progress_bar = QProgressBar()

        # Create the buttons
        ui.start_button = QPushButton("Start and send to XFEL e-LogBook")
        ui.cancel_button = QPushButton("Cancel")

        # Layouts
        spinner_layout = QHBoxLayout()
        spinner_layout.addWidget(ui.images_label)
        spinner_layout.addWidget(ui.image_spinner)
        spinner_layout.addWidget(ui.progress_bar)

        button_layout = QHBoxLayout()
        button_layout.addWidget(ui.start_button)
        button_layout.addWidget(ui.cancel_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(ui.text_edit)
        main_layout.addLayout(spinner_layout)
        main_layout.addLayout(button_layout)

        # Create a central widget, set the layout, and set it as the central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        return ui


def _get_from_screen_instance(screen):
    return screen.get_image()
