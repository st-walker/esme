from dataclasses import dataclass
import logging
from enum import Enum, auto
import time

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QFileDialog, QFrame, QMessageBox, QGridLayout, QWidget
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot, QRunnable
import pyqtgraph as pg
import numpy as np


from esme import DiagnosticRegion
from esme.gui.common import setup_screen_display_widget, send_to_logbook, make_default_i1_lps_machine, make_default_b2_lps_machine
from esme.calibration import TDSCalibration, get_tds_com_slope
from esme.control.tds import StreakingPlane, UncalibratedTDSError


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

pg.setConfigOption("useNumba", True)


# Below this dispersion we take the dispersion to be 0.
# Used for deciding whether to
ZERO_DISPERSION_THRESHOLD = 0.1

# If axis changed by 5% or less then don't propagate a new axis to save unnecessary updates.
AXIS_UPDATE_RELATIVE_TOLERANCE = 0.05


AXES_KWARGS = {"x": {"text": "<i>&Delta;x</i>", "units": "m"},
               "y": {"text": "<i>&Delta;y</i>", "units": "m"},
               "time": {"text": "<i>&Delta;t</i>", "units": "s"},
               "energy": {"text": "<i>&Delta;E</i>", "units": "eV"}}


class ImageTakingMode(Enum):
    BACKGROUND = auto()
    BEAM_IMAGE = auto()
    BEAM_FULL = auto()


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


class ScreenDisplayWidget(QWidget):
    screen_name_signal = pyqtSignal(str)
    tds_calibration_signal = pyqtSignal(object)
    
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.glwidget = pg.GraphicsLayoutWidget(parent=self)
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.glwidget)
        self.image_plot = setup_screen_display_widget(self.glwidget, axes=True)

        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

        self.screen_worker, self.screen_thread = self.setup_screen_worker()
        self.calibration_worker, self.calibration_thread = self.setup_calibration_worker()

        self.tds_calibration_signal.connect(self.calibration_worker.update_tds_voltage_calibration)

        self.add_transverse_projection("x")
        self.add_transverse_projection("y")

        self.set_screen("OTRC.55.I1")

    def propagate_tds_calibration_signal(self, calib):
        print("received a TDS calibration signal...")
        self.tds_calibration_signal.emit(calib)
        self.calibration_worker.update_tds_voltage_calibration(calib)

    def set_screen(self, screen_name: str) -> None:
        self.calibration_worker.screen_name = screen_name
        # Screen dimensions should always be distance (not pixels, not time, not energy).
        dx, dy = self.machine.optics.dispersions_at_screen(screen_name)

        screen = self.machine.screens[screen_name]
        xpixel_size = screen.get_pixel_xsize()
        ypixel_size = screen.get_pixel_ysize()

        nxpixel = screen.get_image_xpixels()
        nypixel = screen.get_image_ypixels()

        tr = QtGui.QTransform()  # prepare ImageItem transformation:
        tr.scale(xpixel_size, ypixel_size)       # scale horizontal and vertical axes
        tr.translate(-nypixel/2, -nxpixel/2) # move 3x3 image to locate center at axis origin
        self.image_plot.items[0].setTransform(tr) # assign transform

    @pyqtSlot(ImagePayload)
    def post_beam_image(self, image_payload):
        items = self.image_plot.items
        assert len(items) == 1
        image_item = items[0]
        # Flip lr is necessary to make it look good, this goes in
        # combination with the imageAxisOrder of pyqtgraph!  Changing
        # one requires the other to also change...
        image_item.setImage(np.fliplr(image_payload.image))

    def set_axis_transform_with_label(self, axis_calib):
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

    def add_transverse_projection(self, dimension):
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

    def setup_screen_worker(self):
        LOG.debug("Initialising screen worker thread")
        screen_worker = ScreenWatcher(self.machine)
        self.screen_name_signal.connect(screen_worker.set_screen_name)
        screen_thread = QThread()
        screen_worker.moveToThread(screen_thread)
        screen_thread.started.connect(screen_worker.run)
        screen_thread.start() # XXX Is this important?!!??!?!  Before signal connection?!?
        screen_worker.image_signal.connect(self.post_beam_image)
        screen_worker.image_signal.connect(self.update_projection_plots)

        return screen_worker, screen_thread

    def setup_calibration_worker(self):
        LOG.debug("Initialising calibration worker thread")
        # XXX need to update machine when I1 or B2 is changed!

        calib_worker = CalibrationWatcher(self.machine, "OTRC.55.I1")
        self.screen_name_signal.connect(calib_worker.set_screen_name)

        calib_thread = QThread()
        calib_worker.moveToThread(calib_thread)
        calib_thread.started.connect(calib_worker.run)
        calib_worker.axis_calibration_signal.connect(self.set_axis_transform_with_label)
        calib_thread.start()
        return calib_worker, calib_thread

    def closeEvent(self, event):
        self.screen_worker.kill = True
        self.calibration_worker.kill = True
        self.screen_thread.terminate()
        self.screen_thread.wait()
        self.calibration_thread.terminate()
        self.calibration_thread.wait()
    
    def update_projection_plots(self, image_payload):
        image = np.fliplr(image_payload.image)
        sh = image.shape

        xpixel_size = image_payload.pixels.xsize
        ypixel_size = image_payload.pixels.ysize
        import scipy
        # t_sg = scipy.signal.savgol_filter(t, window_length=35, polyorder=2)
        xproj = image.sum(axis=1, dtype=np.float64)
        # print(sum(xproj))
        # xproj /= np.sum(xproj)
        xproj = scipy.signal.savgol_filter(xproj, window_length=20, polyorder=2)
        yproj = image.sum(axis=0, dtype=np.float64)
        # yproj /= np.sum(yproj)

        yproj = scipy.signal.savgol_filter(yproj, window_length=20, polyorder=2)

        x = np.linspace(-sh[0]/2, sh[0]/2, num=len(xproj)) * xpixel_size  # ? Why?
        y = np.linspace(-sh[1]/2, sh[1]/2, num=len(yproj)) * ypixel_size

        self.xplot.plot(x, xproj)
        self.yplot.plot(yproj, y)

    

class CalibrationWatcher(QObject):
    # This is necessary in case the calibrations change at all at the
    # screen.  E.g. energy changes, position changes, dispersion changes..  etc.

    # XXX: What if scale factor is 0?
    axis_calibration_signal = pyqtSignal(AxisCalibration)
    def __init__(self, machine, screen_name):
        super().__init__()

        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

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
            self.check_non_streaking_plane_axis_calibration()
            self.check_streaking_plane_axis_calibration()
            time.sleep(0.25)

    def set_screen_name(self, screen_name: str) -> None:
        # necessary to have this method for connecting signal to.
        self.screen_name = screen_name

        


class ScreenWatcher(QObject):
    image_signal = pyqtSignal(object)
    pixels_signal = pyqtSignal(PixelInfo)
    background_signal = pyqtSignal(object)

    def __init__(self, machine):
        super().__init__()
        self.machine = machine
        self.screen_name = "OTRC.55.I1"
        self.kill = False
        self.mode = ImageTakingMode.BACKGROUND
        self.set_screen_name(self.screen_name)

    def get_image(self):
        image = self.machine.screens[self.screen_name].get_image()
        LOG.info("Reading image from: %s", self.screen_name)
        return ImagePayload(image=image, pixels=self.pixels)

    def run(self):
        while not self.kill:
            time.sleep(1)
            image = self.get_image()
            if image is None:
                continue
            else:
                self.image_signal.emit(image)

    def set_screen_name(self, screen_name):
        LOG.info(f"Setting screen name for Screen Worker thread: {screen_name}")
        self.screen_name = screen_name
        screen = self.machine.screens[screen_name]
        xsize = screen.get_pixel_xsize()
        ysize = screen.get_pixel_ysize()
        nx = screen.get_image_xpixels()
        ny = screen.get_image_ypixels()

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
            out = []
            strings = [str(-float(s)) for s in strings]
        return strings
    
        
