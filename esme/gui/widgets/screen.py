from __future__ import annotations

from dataclasses import dataclass
import logging
from enum import Enum, auto
import time
from collections import deque
import tarfile
from io import BytesIO

from PyQt5 import QtGui
from PyQt5.QtWidgets import QGridLayout, QWidget
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, QTimer
import pyqtgraph as pg
import numpy as np
import numpy.typing as npt
import scipy
import pandas as pd

from esme import DiagnosticRegion
from .common import get_machine_manager_factory, setup_screen_display_widget
from esme.calibration import get_tds_com_slope
from esme.control.tds import StreakingPlane, UncalibratedTDSError
from esme.control.exceptions import DOOCSReadError
from esme.control.machines import MachineReadManager
from esme.control.screens import Screen

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

pg.setConfigOption("useNumba", True)


# Below this dispersion we take the dispersion to be 0.
# Used for deciding whether to apply an energy axis to the screen or not.
ZERO_DISPERSION_THRESHOLD = 0.1

# If axis changed by 5% or less then don't propagate a new axis to save unnecessary updates.
AXIS_UPDATE_RELATIVE_TOLERANCE = 0.05


AXES_KWARGS = {"x": {"text": "<i>&Delta;x</i>", "units": "m"},
               "y": {"text": "<i>&Delta;y</i>", "units": "m"},
               "time": {"text": "<i>&Delta;t</i>", "units": "s"},
               "energy": {"text": "<i>&Delta;E</i>", "units": "eV"}}


class ImageTakingMode(Enum):
    BACKGROUND = auto()
    BEAM = auto()


@dataclass
class AxisCalibration:
    scale_factor: float
    parameter: str # x, y, time, energy
    axis: str # X or Y


@dataclass
class PixelInfo:
    xsize: float
    ysize: float
    nx: int
    ny: int


@dataclass
class ImagePayload:
    image: np.ndarray
    pixels: PixelInfo
    levels: tuple[float, float]
    xproj: np.ndarray
    yproj: np.ndarray

    @property
    def x(self) -> np.ndarray:
        sh = self.image.shape
        return np.linspace(-sh[0]/2, sh[0]/2, num=len(self.xproj)) * self.pixels.xsize

    @property
    def y(self) -> np.ndarray:
        sh = self.image.shape
        return np.linspace(-sh[1]/2, sh[1]/2, num=len(self.yproj)) * self.pixels.ysize


class ScreenDisplayWidget(QWidget):
    screen_name_signal = pyqtSignal(str)
    tds_calibration_signal = pyqtSignal(object)

    # Signals for interacting with the daughter workers in their respective threads.
    save_to_elog_signal = pyqtSignal()
    take_background_signal = pyqtSignal()
    subtract_background_signal = pyqtSignal(bool)


    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.glwidget = pg.GraphicsLayoutWidget(parent=self)
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.glwidget)
        self.image_plot = setup_screen_display_widget(self.glwidget, axes=True)

        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine # Set initial machine choice to be for I1 diagnostics

        self.data_worker, self.data_thread = self.setup_data_taking_worker()
        self.calibration_worker, self.calibration_thread = self.setup_calibration_worker()

        self.tds_calibration_signal.connect(self.calibration_worker.update_tds_voltage_calibration)

        # Add projection axes to the image display daughter widget
        self.add_transverse_projection("x")
        self.add_transverse_projection("y")

        # We have to pick some screen to begin with so it might as well be the first screen.
        self.set_screen("OTRC.55.I1")

    def propagate_tds_calibration_signal(self, calib) -> None:
        print("received a TDS calibration signal...")
        self.tds_calibration_signal.emit(calib)
        self.calibration_worker.update_tds_voltage_calibration(calib)

    def set_screen(self, screen_name: str) -> None:
        self.calibration_worker.screen_name = screen_name
        self.data_worker.screen_name = screen_name
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
        tr.scale(xpixel_size, ypixel_size)       # scale horizontal and vertical axes
        tr.translate(-nypixel/2, -nxpixel/2) # move 3x3 image to locate center at axis origin
        self.image_plot.items[0].setTransform(tr) # assign transform

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
            axis = {"right": NegatableLabelsAxisItem("right", text="<i>&Delta;y</i>", units="m")}
            self.yplot = win.addPlot(row=0, col=1, rowspan=1, colspan=1, axisItems=axis)
            self.yplot.hideAxis("left")
            self.yplot.hideAxis("bottom")
            self.yplot.getViewBox().invertX(True)
            self.yplot.setMaximumWidth(200)
            self.yplot.setYLink(self.image_plot)
        elif dimension == "x":
            # Another plot area for displaying ROI data
            win.nextRow()
            axis = {"bottom": NegatableLabelsAxisItem("bottom", text="<i>&Delta;x</i>", units="m")}
            self.xplot = win.addPlot(row=1, col=0, colspan=1, axisItems=axis)
            self.xplot.hideAxis("left")
            self.xplot.setMaximumHeight(200)
            self.xplot.setXLink(self.image_plot)

    def setup_data_taking_worker(self) -> tuple[DataTakingWorker, QThread]:
        LOG.debug("Initialising screen worker thread")
        data_worker = DataTakingWorker()
        # Propagate screen name change to data taking worker
        # responsible for reading from the screen.
        self.screen_name_signal.connect(data_worker.set_screen)
        data_thread = QThread()
        data_thread.started.connect(data_worker.start_timers)
        data_thread.finished.connect(data_worker.stop_timers)

        self.save_to_elog_signal.connect(data_worker.dump_state_to_file)
        self.take_background_signal.connect(data_worker.take_background)
        self.subtract_background_signal.connect(data_worker.set_subtract_background)
 
        data_worker.display_image_signal.connect(self.post_beam_image)
        data_worker.display_image_signal.connect(self.update_projection_plots)
        data_worker.moveToThread(data_thread)
        data_thread.start()

        return data_worker, data_thread



    def setup_calibration_worker(self) -> tuple[CalibrationWatcher, QThread]:
        LOG.debug("Initialising calibration worker thread")
        # XXX need to update machine when I1 or B2 is changed!

        calib_worker = CalibrationWatcher(self.machine, "OTRC.55.I1")
        self.screen_name_signal.connect(calib_worker.set_screen_name)

        calib_thread = QThread()
        calib_worker.axis_calibration_signal.connect(self.set_axis_transform_with_label)
        calib_thread.started.connect(calib_worker.run)
        calib_worker.moveToThread(calib_thread)
        # NOTE: NOT STARTING THREAD FOR NOW BECAUSE IT IS A HUGE BOTTLENECK AND SLOWS DOWN GUI TOO MUCH!
        # calib_thread.start()
        return calib_worker, calib_thread

    def closeEvent(self, event) -> None:
        self.data_worker.kill = True
        self.calibration_worker.kill = True
        self.data_thread.terminate()
        self.data_thread.wait()
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

        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine # Set initial machine choice to be for I1 diagnostics

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
        scale_factor = 1
        parameter = axis

        try:
            voltage = self.machine.deflector.get_voltage_rb()
            r12 = self.machine.optics.r12_streaking_from_tds_to_point(self.screen_name)
            energy_mev = self.machine.optics.get_beam_energy() # MeV is fine here.
            com_slope = get_tds_com_slope(r12, energy_mev, voltage)
            scale_factor = 1 / com_slope
            parameter = "time"
        except UncalibratedTDSError:
            pass

        return AxisCalibration(scale_factor=scale_factor,
                               parameter=parameter,
                               axis=axis)

    def get_non_streaking_plane_calibration(self) -> AxisCalibration:
        beam_energy = self.machine.optics.get_beam_energy() * 1e6 # in MeV to eV

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

        return AxisCalibration(scale_factor=scale_factor, parameter=parameter, axis=axis)

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
        if axis_calibrations_are_substantially_different(calib,
                                                         self.non_streaking_plane_calibration):
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
        if axis_calibrations_are_substantially_different(calib,
                                                         self.streaking_plane_calibration):
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


class DataTakingWorker(QObject):
    display_image_signal = pyqtSignal(ImagePayload)
    pixels_signal = pyqtSignal(PixelInfo)

    def __init__(self, initial_screen: str = "OTRC.55.I1"):
        super().__init__()

        mfactory: MachineReadManager = get_machine_manager_factory()
        self.i1reader: MachineReadManager = mfactory.make_machine_reader_manager(DiagnosticRegion.I1)
        self.b2reader: MachineReadManager = mfactory.make_machine_reader_manager(DiagnosticRegion.B2)
        self.mreader = self.i1reader # Set initial machine reader instance choice to be for I1.

        # The background images cache.
        self.bg_images: deque[npt.ArrayLike] = deque(maxlen=5)
        # The cache of data we have taken.  We are constantly updating
        self.beam_images: deque[npt.ArrayLike] = deque(maxlen=20)

        self.mode = ImageTakingMode.BEAM
        self.screen: Screen | None = None
        self.pixels: PixelInfo | None = None
        self.set_screen(initial_screen)
        self._do_subtract_background: bool = False

        hardcoded_path = "/Users/xfeloper/user/stwalker/lps-dumpage/"
        self.outdir = hardcoded_path

        # Read from the machine regularly, at the rep. rate of the machine.
        self.beam_images_timer = self.make_screen_read_timer(interval_ms=100)
        # Emit an image (intended for display) only once ever 1s.  There is no need
        # for more frequently than that really.
        self.display_timer = self.make_display_timer(interval_ms=1000)

    def set_region(self, region: DiagnosticRegion) -> None:
        if region is DiagnosticRegion.I1:
            self.mreader = self.i1reader
        elif region is DiagnosticRegion.B2:
            self.mreader = self.b2reader
        else:
            raise ValueError(f"Unknown region enum:, {region}")

    def take_background(self) -> None:
        # Set mode to background taking.
        self.mode = ImageTakingMode.BACKGROUND
        self._clear_bg_cache()

    def set_subtract_background(self, subtract_bg_state: Qt.CheckState) -> None:
        self._do_subtract_background = bool(subtract_bg_state)

    def dump_state_to_file(self) -> None:
        kvps = self.machine.full_read()
        kvps = pd.Series(kvps)
        kvps.index.name = "channel"
        kvps.reset_index(name="value")

        # with tarfile.open(outdir, "w:gz") as tarball:  # With compression
        with tarfile.open(self.outdir, "w") as tarball: # No compression for now
            # Pickle is about 5x faster but for now prefer np.savez because I
            # don't know what the long term implications are for using pickle
            # e.g. will some file still be readable in a few year's time?
            buffer = BytesIO()
            np.savez(buffer,
                     beam_images=self.beam_images,
                     bg_images=self.bg_images)
            buffer.seek(0)
            images_tarinfo = tarfile.TarInfo(name=f"{outdir}/images.npz")
            tarball.addfile(images_tarinfo, fileobj=buffer)
            buffer.seek(0)
            # Now write all the various DOOCS addresses we've read to file
            kvps_tarinfo = tarfile.TarInfo(name=f"{outdir}/channels.csv")
            kvps.to_csv(buffer, index=False)
            tarball.addfile(kvps_tarinfo, fileobj=buffer)

            # XXX TDS CALIBRATION NEEDS TO SOMEHOW BE INCLUDED HERE !!

            # XXX ALSO SCREEN NAME!!

    def start_timers(self) -> None:
        self.beam_images_timer.start()
        self.display_timer.start()

    def stop_timers(self) -> None:
        self.beam_images_timer.stop()
        self.display_timer.stop()

    def make_screen_read_timer(self, interval_ms: int) -> QTimer:
        timer = QTimer()
        timer.setInterval(interval_ms)
        timer.timeout.connect(self.read_from_screen)
        return timer

    def make_display_timer(self, interval_ms: int) -> QTimer:
        timer = QTimer()
        timer.setInterval(interval_ms)
        timer.timeout.connect(self.process_and_emit_image_for_display)
        return timer

    def _clear_caches(self):
        self._clear_bg_cache()
        self.beam_images = deque(maxlen=self.beam_images.maxlen)

    def _clear_bg_cache(self):
        self.bg_images = deque(maxlen=self.bg_images.maxlen)
        
    def process_and_emit_image_for_display(self) -> None:
        # Get most images, which for a deque where we are appending left,
        # we want the 0th element.
        try:
            image = self.beam_images[0]
        except IndexError:
            # We haven't acquired any images yet.  Do nothing.
            return

        # Subtract background if we have taken some background and enabled it.
        if self.bg_images and self._subtract_background:
            mean_bg = np.mean(self.bg_images, axis=0)
            image -= mean_bg
            
        minpix = image.min()
        maxpix = image.max()
        xproj = image.sum(axis=1, dtype=np.float32)
        # xproj = scipy.signal.savgol_filter(xproj, window_length=20, polyorder=2)
        yproj = image.sum(axis=0, dtype=np.float32)

        imp = ImagePayload(image=image,
                           pixels=self.pixels,
                           levels=(minpix, maxpix),
                           xproj=xproj,
                           yproj=yproj)

        self.display_image_signal.emit(imp)

    def read_from_screen(self) -> None:
        image = self.screen.get_image()
        if self.mode is ImageTakingMode.BEAM:
            self.beam_images.appendleft(image)
        elif self.mode is ImageTakingMode.BACKGROUND:
            self.bg_images.appendleft(image)
            # Then we've filled the ring buffer of background images and
            # can go back to taking beam data...
            if len(self.bg_images) == self.bg_images.maxlen:
                self.mode = ImageTakingMode.BEAM
        else:
            raise ValueError(f"Unknown image taking mode: {self.mode}")

    def set_screen(self, screen_name: str) -> None:
        LOG.info(f"Setting screen name for Screen Worker thread: {screen_name}")
        screen = self.mreader.screens[screen_name]
        self.screen = screen
        self._clear_caches()
        try:
            xsize = screen.get_pixel_xsize()
            ysize = screen.get_pixel_ysize()
            nx = screen.get_image_xpixels()
            ny = screen.get_image_ypixels()
        except DOOCSReadError as e:
            return

        pix = PixelInfo(xsize=xsize, ysize=ysize, nx=nx, ny=ny)
        self.pixels_signal.emit(pix)
        self.pixels = pix

def axis_calibrations_are_substantially_different(calib1: AxisCalibration, calib2: AxisCalibration) -> bool:
    if (calib1 is None) ^ (calib2 is None):
        return True

    # (calib1 is None) or (calib2 is None)

    are_identical = calib1 is calib2
    parameters_are_different = calib1.parameter != calib2.parameter
    axis_are_different = calib1.axis != calib2.axis
    scales_are_close_enough = np.isclose(calib1.scale_factor,  # if transformations are close enough (to avoid small updates)
                                         calib2.scale_factor,
                                         rtol=AXIS_UPDATE_RELATIVE_TOLERANCE)
    if are_identical:
        return False

    if parameters_are_different:
        return True

    if axis_are_different:
        return True

    if not scales_are_close_enough:
        return False


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
