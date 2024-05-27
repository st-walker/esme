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
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import (
    QGridLayout,
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
from esme.calibration import get_tds_com_slope
from esme.control.exceptions import DOOCSReadError
from esme.control.machines import MachineReadManager
from esme.control.screens import Screen, ScreenMetadata
from esme.control.tds import StreakingPlane, UncalibratedTDSError
from esme.gui.ui.imaging import Ui_imaging_widget

from .common import (
    get_machine_manager_factory,
    send_widget_to_log,
    setup_screen_display_widget,
)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

pg.setConfigOption("useNumba", True)


# Below this dispersion we take the dispersion to be 0.
# Used for deciding whether to apply an energy axis to the screen or not.
ZERO_DISPERSION_THRESHOLD = 0.1

# If axis changed by 5% or less then don't propagate a new axis to save unnecessary updates.
AXIS_UPDATE_RELATIVE_TOLERANCE = 0.05


AXES_KWARGS = {
    "x": {"text": "<i>&Delta;x</i>", "units": "m"},
    "y": {"text": "<i>&Delta;y</i>", "units": "m"},
    "time": {"text": "<i>&Delta;t</i>", "units": "s"},
    "energy": {"text": "<i>&Delta;E</i>", "units": "eV"},
}


class MessageType(Enum):
    CACHE_BACKGROUND = auto()
    # TAKE_N_FAST = auto()
    CHANGE_SCREEN = auto()
    SUBTRACT_BACKGROUND = auto()


@dataclass
class DataTakingMessage:
    mtype: MessageType
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AxisCalibration:
    scale_factor: float
    parameter: str  # x, y, time, energy
    axis: str  # X or Y


@dataclass
class ImagePayload:
    image: np.ndarray
    screen_md: ScreenMetadata
    levels: tuple[float, float]
    xproj: np.ndarray
    yproj: np.ndarray

    @property
    def x(self) -> np.ndarray:
        sh = self.image.shape
        return (
            np.linspace(-sh[0] / 2, sh[0] / 2, num=len(self.xproj))
            * self.screen_md.xsize
        )

    @property
    def y(self) -> np.ndarray:
        sh = self.image.shape
        return (
            np.linspace(-sh[1] / 2, sh[1] / 2, num=len(self.yproj))
            * self.screen_md.ysize
        )


class StopImageAcquisition(Exception):
    pass


@dataclass
class MachineState:
    bg_images: deque[npt.NDArray]
    kvps: pd.Series


class ImagingControlWidget(QWidget):
    """The Image control widget is the widget that gets the image and
    then possibly pushes it to the screen widget.

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


class ScreenWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.glwidget = pg.GraphicsLayoutWidget(parent=self)
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.glwidget)
        self.image_plot = setup_screen_display_widget(self.glwidget, axes=True)

        # fmt: off
        self.i1machine, self.b2machine, get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine
        self.calibration_worker, self.calibration_thread = self.setup_calibration_worker()
        # fmt: on

        self.tds_calibration_signal.connect(
            self.calibration_worker.update_tds_voltage_calibration
        )

        # Add projection axes to the image display daughter widget
        self.add_transverse_projection("x")
        self.add_transverse_projection("y")

        # We have to pick some screen to begin with so it might as well be the first screen.
        self.set_screen("OTRC.55.I1")

    def propagate_tds_calibration_signal(self, calib) -> None:
        self.calibration_worker.update_tds_voltage_calibration(calib)

    def set_screen(self, screen_name: str) -> None:
        self.calibration_worker.screen_name = screen_name
        # Screen dimensions should always be distance (not pixels, not time, not energy).
        dx, dy = self.machine.optics.dispersions_at_screen(screen_name)

        screen = self.machine.screens[screen_name]
        # XXX

        # If the camera is not switched on then these reads will fail.
        # but this shouldn't simply kill the whole gui.
        try:
            xpixel_size = screen.get_pixel_xsize()
            ypixel_size = screen.get_pixel_ysize()

            nxpixel = screen.get_image_xpixels()
            nypixel = screen.get_image_ypixels()
        except DOOCSReadError as e:
            return

        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.scale(xpixel_size, ypixel_size)  # scale horizontal and vertical axes
        tr.translate(
            -nypixel / 2, -nxpixel / 2
        )  # move 3x3 image to locate center at axis origin
        self.image_plot.items[0].setTransform(tr)  # assign transform

    @pyqtSlot(ImagePayload)
    def post_beam_image(self, image_payload: ImagePayload) -> None:
        items = self.image_plot.items
        assert len(items) == 1
        image_item = items[0]
        # Flip lr is necessary to make it look good, this goes in
        # combination with the imageAxisOrder of pyqtgraph!  Changing
        # one requires the other to also change...
        image_item.setLevels(image_payload.levels)
        image_item.setImage(image_payload.image)

    def set_axis_transform_with_label(self, axis_calib: AxisCalibration) -> None:
        if axis_calib.axis == "x":
            axis = self.xplot.getAxis("bottom")
        elif axis_calib.axis == "y":
            axis = self.yplot.getAxis("right")
        # Could be x, y or time in streaking plane.
        # or x, y or energy in non streaking plane.
        axis.setLabel(**AXES_KWARGS[axis_calib.parameter])
        axis.negate = axis_calib.scale_factor < 0
        axis.setScale(abs(axis_calib.scale_factor))

        self.xplot.update()

    def add_transverse_projection(self, dimension: str) -> None:
        win = self.glwidget
        if dimension == "y":
            axis = {
                "right": NegatableLabelsAxisItem(
                    "right", text="<i>&Delta;y</i>", units="m"
                )
            }
            self.yplot = win.addPlot(row=0, col=1, rowspan=1, colspan=1, axisItems=axis)
            self.yplot.hideAxis("left")
            self.yplot.hideAxis("bottom")
            self.yplot.getViewBox().invertX(True)
            self.yplot.setMaximumWidth(200)
            self.yplot.setYLink(self.image_plot)
        elif dimension == "x":
            # Another plot area for displaying ROI data
            win.nextRow()
            axis = {
                "bottom": NegatableLabelsAxisItem(
                    "bottom", text="<i>&Delta;x</i>", units="m"
                )
            }
            self.xplot = win.addPlot(row=1, col=0, colspan=1, axisItems=axis)
            self.xplot.hideAxis("left")
            self.xplot.setMaximumHeight(200)
            self.xplot.setXLink(self.image_plot)

    def setup_calibration_worker(self) -> tuple[CalibrationWatcher, QThread]:
        LOG.debug("Initialising calibration worker thread")
        # XXX need to update machine when I1 or B2 is changed!

        calib_worker = CalibrationWatcher(self.machine, "OTRC.55.I1")
        calib_thread = QThread()
        calib_worker.moveToThread(calib_thread)
        calib_worker.axis_calibration_signal.connect(self.set_axis_transform_with_label)
        calib_thread.started.connect(calib_worker.run)
        # NOTE: NOT STARTING THREAD FOR NOW BECAUSE IT IS A HUGE BOTTLENECK AND SLOWS DOWN GUI TOO MUCH!
        # XXXXXXXXXXXXXXXX:
        # calib_thread.start()
        return calib_worker, calib_thread

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.calibration_worker.kill = True
        self.calibration_thread.terminate()
        self.calibration_thread.wait()

    def update_projection_plots(self, image_payload: ImagePayload) -> None:
        self.xplot.clear()
        self.yplot.clear()
        self.xplot.plot(image_payload.x, image_payload.xproj)
        self.yplot.plot(image_payload.yproj, image_payload.y)


class CalibrationWatcher(QObject):
    # This is necessary in case the calibrations change at all at the
    # screen.  E.g. energy changes, position changes, dispersion changes..  etc.

    # XXX: What if scale factor is 0?
    axis_calibration_signal = pyqtSignal(AxisCalibration)

    def __init__(self, machine, screen_name: str):
        super().__init__()

        (
            self.i1machine,
            self.b2machine,
        ) = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = (
            self.i1machine
        )  # Set initial machine choice to be for I1 diagnostics

        self.screen_name = screen_name
        self.kill = False

        # Initially no axis calibrations at all.
        self.streaking_plane_calibration = None
        self.non_streaking_plane_calibration = None

    def update_tds_voltage_calibration(self, tdscalib):
        print(tdscalib, "..................>????????????????????")
        region = DiagnosticRegion(tdscalib.region)
        if region is DiagnosticRegion.I1:
            self.i1machine.deflector.calibration = tdscalib
        elif region is DiagnosticRegion.B2:
            self.b2machine.deflector.calibration = tdscalib

    def get_streaking_plane_calibration(self) -> AxisCalibration:
        axis = self.get_streaking_plane()
        # If TDS is set to be used in the special bunch mid layer then
        # we assume a streaked beam and we would like to have a time
        # axis:
        # if self.machine.sbunches.get_use_tds():
        scale_factor = 1.0
        parameter = axis

        try:
            voltage = self.machine.deflector.get_voltage_rb()
            r12 = self.machine.optics.r12_streaking_from_tds_to_point(self.screen_name)
            energy_mev = self.machine.optics.get_beam_energy()  # MeV is fine here.
            com_slope = get_tds_com_slope(r12, energy_mev, voltage)
            scale_factor = 1.0 / com_slope
            parameter = "time"
        except UncalibratedTDSError:
            pass

        return AxisCalibration(
            scale_factor=scale_factor, parameter=parameter, axis=axis
        )

    def get_non_streaking_plane_calibration(self) -> AxisCalibration:
        beam_energy = self.machine.optics.get_beam_energy() * 1e6  # in MeV to eV

        axis = self.get_non_streaking_plane()

        # If there is no dispersion then we just plot x or y
        # (depending on which plane is non-streaking plane).
        if self.is_dispersive_at_screen():
            dispersion = self.get_dispersion_at_screen()
            scale_factor = beam_energy / dispersion
            parameter = "energy"
        else:
            scale_factor = 1
            parameter = axis

        return AxisCalibration(
            scale_factor=scale_factor, parameter=parameter, axis=axis
        )

    def is_dispersive_at_screen(self) -> bool:
        dispersion = self.get_dispersion_at_screen()
        return abs(dispersion) > ZERO_DISPERSION_THRESHOLD

    def get_streaking_plane(self) -> str:
        plane = self.machine.deflector.plane
        if plane is StreakingPlane.HORIZONTAL:
            return "x"
        elif plane is StreakingPlane.VERTICAL:
            return "y"
        else:
            raise ValueError(f"Unknown TDS streaking plane: {plane}")

    def get_non_streaking_plane(self) -> str:
        sp = self.get_streaking_plane()
        if sp == "x":
            return "y"
        return "x"

    def get_dispersion_at_screen(self) -> float:
        dx, dy = self.machine.optics.dispersions_at_screen(self.screen_name)
        # Return whichever is bigger.  This is fine because dispersion
        # only ever in a single plane in the dumplines, either x (in I1D) or y (B2D).
        if abs(dx) > abs(dy):
            return dx
        return dy

    def check_non_streaking_plane_axis_calibration(self):
        """Calculate the axis calibration for the non-streaking plane.
        Could be x, y or energy depending on where we are in the machine.

        Will only be energy if in dispersive section.

        Otherwise could be x or y if in I1 (x) or y (B2)

        """
        calib = self.get_non_streaking_plane_calibration()
        if axis_calibrations_are_substantially_different(
            calib, self.non_streaking_plane_calibration
        ):
            self.non_streaking_plane_calibration = calib
            self.axis_calibration_signal.emit(calib)

    def check_streaking_plane_axis_calibration(self):
        """Calculate the axis calibration for the streaking plane.

        Will only be time if the TDS has been calibrated.

        Otherwise could be x or y (if in I1 (y) or B2 (x).

        Could be x, y or time depending on whether or not the TDS has
        been calibrated (either time or not) or x or y (if we are in
        BC2 or I1).

        """

        calib = self.get_streaking_plane_calibration()
        if axis_calibrations_are_substantially_different(
            calib, self.streaking_plane_calibration
        ):
            self.streaking_plane_calibration = calib
            self.axis_calibration_signal.emit(calib)

    def run(self) -> None:
        while not self.kill:
            # Be tolerant of DOOCSReadErrors which might be very fleeting
            # and not really a problem, e.g.
            # if beam goes off for a moment we won't be able to read the energy
            # for a bit, but that shoudln't crash the whole GUI.
            try:
                self.check_non_streaking_plane_axis_calibration()
                self.check_streaking_plane_axis_calibration()
            except DOOCSReadError:
                pass
            time.sleep(0.25)

    def set_screen_name(self, screen_name: str) -> None:
        # necessary to have this method for connecting signal to.
        self.screen_name = screen_name


def axis_calibrations_are_substantially_different(
    calib1: AxisCalibration, calib2: AxisCalibration
) -> bool:
    if (calib1 is None) ^ (calib2 is None):
        return True

    # (calib1 is None) or (calib2 is None)

    are_identical = calib1 is calib2
    parameters_are_different = calib1.parameter != calib2.parameter
    axis_are_different = calib1.axis != calib2.axis
    scales_are_close_enough = np.isclose(
        calib1.scale_factor,  # if transformations are close enough (to avoid small updates)
        calib2.scale_factor,
        rtol=AXIS_UPDATE_RELATIVE_TOLERANCE,
    )
    if are_identical:
        return False

    if parameters_are_different:
        return True

    if axis_are_different:
        return True

    if not scales_are_close_enough:
        return False

    raise ValueError("Unable to understand what is going on here")


class NegatableLabelsAxisItem(pg.AxisItem):
    """This is used for where I flip the positive/negative direction.
    Positive x is to the right but if we have negative dispersion,
    positive energy is to the left, for example, so this is what this
    is for.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negate = False

    def tickStrings(self, values, scale, spacing):
        # should probably do this in tickValues instead, but this is waye easier.
        # Zero ends up signed which I don't like but oh well.
        strings = super().tickStrings(values, scale, spacing)
        if self.negate:
            strings = [str(-float(s)) for s in strings]
        return strings


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

        self.bg_images = bg_images
        self._executor = ProcessPoolExecutor(max_workers=1)

    def set_screen(self, screen: Screen):
        self._screen = screen

    def _connect_buttons(self):
        # Connect cancel button to close the window
        self.ui.cancel_button.clicked.connect(self.close)
        self.ui.start_button.clicked.connect(self.start)

    def start(self) -> None:
        self.ui.progress_bar.setMaximum(self.ui.image_spinner.value())
        # Get images from data taking thread
        images = self._acquire_n_images_fast_in_subprocess(
            self.ui.image_spinner.value()
        )

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
            np.savez(buffer, beam_images=beam_images, bg_images=mstate.bg_images)
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

        ui.setWindowTitle("XFEL e-LogBook Writer")

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
        ui.setCentralWidget(central_widget)

        return ui


def _get_from_screen_instance(screen):
    return screen.get_image()
