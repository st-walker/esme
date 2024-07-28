from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QObject, QTimer, pyqtSlot
from PyQt5.QtWidgets import QGridLayout, QWidget, QGraphicsTextItem
from PyQt5.QtGui import QColor

from esme import DiagnosticRegion
from esme.calibration import get_tds_com_slope
from esme.control.exceptions import DOOCSReadError
from esme.control.screens import ScreenMetadata
from esme.control.tds import StreakingPlane, UncalibratedTDSError
from esme.maths import gauss, get_gaussian_fit
from enum import Enum, auto
from functools import cached_property
from esme.gui.workers import ImagePayload
from esme.core import region_from_screen_name

from .common import get_machine_manager_factory, setup_screen_display_widget

# Below this dispersion we take the dispersion to be 0.
# Used for deciding whether to apply an energy axis to the screen or not.
ZERO_DISPERSION_THRESHOLD = 0.01
# ZERO_STREAKING_THRESHOLD = 0.01 #

# If axis changed by 5% or less then don't propagate a new axis to save unnecessary updates.
AXIS_UPDATE_RELATIVE_TOLERANCE = 0.05

AXES_KWARGS: dict[str : dict[str, Any]] = {
    "x": {"text": "<i>&Delta;x</i>", "units": "m"},
    "y": {"text": "<i>&Delta;y</i>", "units": "m"},
    "time": {"text": "<i>&Delta;t</i>", "units": "s"},
    "energy": {"text": "<i>&Delta;E</i>", "units": "eV"},
}


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


    

@dataclass
class AxesCalibration:
    energy_ev: float
    dispersion: float
    time_calibration: float # in m/s of course...
    streaking_plane: StreakingPlane


class ScreenWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # fmt: off
        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_imaging_managers()
        self.machine = self.i1machine
        # fmt: on

        self.glwidget = pg.GraphicsLayoutWidget(parent=self)
        self.layout = QGridLayout(self)
        self.layout.addWidget(self.glwidget)
        self.image_plot = setup_screen_display_widget(self.glwidget)
        self.setup_screen_axes()

        # Add projection axes to the image display daughter widget
        self.xplot, self.yplot = self._make_transverse_projections()
        # self.gauss_beam_length_string = pg.TextItem(html=f"<i>σ</i><sub>Gauss.</sub> = ", anchor=(0.0, 0), color="white")
        # self.gauss_beam_length_string = pg.TextItem(f"HELLO ", color="yellow")
        self.gauss_beam_length_string = pg.TextItem("text", color="yellow")
        # self.gauss_beam_length_string.setFlag(self.gauss_beam_length_string.ItemIgnoresTransformations)
        self.image_plot.addItem(self.gauss_beam_length_string, ignoreBounds=True)
        self.gauss_beam_length_string.setParentItem(self.image_plot)
        self.gauss_beam_length_string.setPos(1000, 650)

        # self.gauss_beam_length_string.setHtml("<i>Hello</i>")
        # self.xplot.addItem(self.gauss_beam_length_string)
        # self.yplot.addItem(self.gauss_beam_length_string)
        # self.xplot.addItem(self.gauss_beam_length_string)
        # scene = self.glwidget.scene()
        # text_item = QGraphicsTextItem("sample Text")
        # text_item.setDefaultTextColor(QColor("white"))
        # text_item.setPos(600, 750)
        # scene.addItem(text_item)


        # We have to pick some screen to begin with so it might as well be the first screen.
        self.set_screen("OTRC.55.I1")

    def lock_aspect_ratio(self, *, do_lock: bool) -> None:
        self.image_plot.setAspectLocked(do_lock)

    def setup_screen_axes(self) -> None:
        xlabel = "<i>&Delta;x</i>"
        ylabel = "<i>&Delta;y</i>"
        if self.machine is self.i1machine:
            xpos = "right"
            ypos = "bottom"
        elif self.machine is self.b2machine:
            xpos = "bottom"
            ypos = "right"
        self.image_plot.setLabel(xpos, xlabel, units="m")
        self.image_plot.setLabel(ypos, ylabel, units="m")

    def set_screen(self, screen_name: str) -> None:
        region = region_from_screen_name(screen_name)
        if region is DiagnosticRegion.I1:
            self.machine = self.i1machine
        elif region is DiagnosticRegion.B2:
            self.machine = self.b2machine
        # Screen dimensions should always be distance (not pixels, not time, not energy).
        # dx, dy = self.machine.optics.dispersions_at_screen(screen_name)
        screen = self.machine.screens[screen_name]
        self.clear_image()
        self.set_image_transform(screen_name)

    def set_image_transform(self, screen_name: str) -> None:
        """This will set the transform for the image based on the image metadata, namely
        scaling the image by the size of the pixels and putting putting the origin in
        the middle of the axes.  If we try to do this whilst the screen is unpowered,
        then these reads can fail, so to protect against this, if we fail, we
        try again in a second or so with QTimer.singleShot."""

        # this functions is only responsible for transforming from pixel units to distance units
        # i.e. basically scaling
        screen = self.machine.screens[screen_name]

        def try_it():
            try:
                xpixel_size = screen.get_pixel_xsize()
                ypixel_size = screen.get_pixel_ysize()
                nxpixel = screen.get_image_xpixels()
                nypixel = screen.get_image_ypixels()
            except DOOCSReadError as e:
                # Try again in a bit
                QTimer.singleShot(1000, try_it)
            else:
                # XXX!!! This will be wrong for BC2!!!  This whole thing needs refactoring...
                # This makes no sense, why scale by ypixel size but translate by nxpixels?  nobody knows...
                tr = QtGui.QTransform()  # prepare ImageItem transformation:
                tr.scale(ypixel_size, xpixel_size)  # scale horizontal and vertical axes
                tr.translate(
                    -nxpixel / 2, -nypixel / 2
                )  # move image to locate center at axis origin
                self.image_plot.items[0].setTransform(tr)  # assign transform

        try_it()

    def calibrate_axes(self, axescalib: AxesCalibration):
        self._calibrate_dispersive_axes(axescalib)
        self._calibrate_time_axes(axescalib)

    def _calibrate_dispersive_axes(self, axescalib: AxesCalibration) -> None:
        # The dispersive axis the the other axis to the streaking axis.

        # The dispersive axes always is on the y-axis, because we want
        # time on the x-axis.
        axis = self.xplot.getAxis("right")
        if axescalib.streaking_plane is StreakingPlane.HORIZONTAL:
            dim = "y"
        elif axescalib.streaking_plane is StreakingPlane.VERTICAL:
            dim = "x"
        else:
            raise TypeError(f"Unknown streaking plane: {axescalib.streaking_plane}")

        if abs(axescalib.dispersion) < ZERO_DISPERSION_THRESHOLD:
            self.set_axis_transform_with_label(axis, 1.0, AXES_KWARGS[dim])
        else:
            # Dispersion relation is D = x / (delta_P / P0).
            # We delta_P, we have dispersion and we have x (the physics positions on the screen)
            # delta_P = x * reference_energy
            scale_factor = axescalib.energy_ev / axescalib.dispersion

            self.set_axis_transform_with_label(
                axis, scale_factor, AXES_KWARGS["energy"]
            )

    def _calibrate_time_axes(self, axescalib: AxesCalibration) -> None:
        # Time axis is always on the bottom (x-axis)
        axis = self.xplot.getAxis("bottom")
        if axescalib.streaking_plane is StreakingPlane.HORIZONTAL:
            dim = "x"
        elif axescalib.streaking_plane is StreakingPlane.VERTICAL:
            dim = "y"
        else:
            raise TypeError(f"Unknown streaking plane: {axescalib.streaking_plane}")

        if not abs(axescalib.time_calibration):
            self.xplot.hideAxis("left")
            self.set_axis_transform_with_label(axis, 1.0, AXES_KWARGS[dim])
        else:
            self.xplot.showAxis("left")
            scale_factor = 1 / axescalib.time_calibration
            self.set_axis_transform_with_label(
                axis, scale_factor, AXES_KWARGS["time"]
            )


    def set_axis_transform_with_label(
        self,
        axis: NegatableLabelsAxisItem,
        scale_factor: float,
        axis_kwargs: dict[str, dict[str, Any]],
    ) -> None:
        axis.setLabel(**axis_kwargs)
        axis.negate = scale_factor < 0
        axis.setScale(abs(scale_factor))

    def clear_image(self) -> None:
        self.image_plot.items[0].clear()     
        self.image_plot.items[0].setImage()   

    @pyqtSlot(ImagePayload)
    def post_beam_image(self, image_payload: ImagePayload) -> None:
        items = self.image_plot.items
        image_item = items[0]
        image_item.setImage(image_payload.image)
        self.update_projection_plots(image_payload)

    def _make_transverse_projections(self) -> None:
        win = self.glwidget
        axis = {
            "right": NegatableLabelsAxisItem("right", text="<i>&Delta;x</i>", units="m")
        }
        yplot = win.addPlot(row=0, col=1, rowspan=1, colspan=1, axisItems=axis)
        yplot.hideAxis("left")
        yplot.hideAxis("bottom")
        yplot.getViewBox().invertX(True)
        yplot.setMaximumWidth(200)
        yplot.setYLink(self.image_plot)
        # Another plot area for displaying ROI data
        win.nextRow()
        axis = {
            "bottom": NegatableLabelsAxisItem(
                "bottom", text="<i>&Delta;y</i>", units="m"
            )
        }
        xplot = win.addPlot(row=1, col=0, colspan=1, axisItems=axis)
        xplot.hideAxis("left")
        xplot.setMaximumHeight(200)
        xplot.setXLink(self.image_plot)
        xplot.setLabel("left", text="<i>I</i>", units="A")
        return xplot, yplot

    def update_projection_plots(self, image_payload: ImagePayload) -> None:
        self.xplot.clear()
        self.yplot.clear()
        # We need here the time calibration really (if it exists...)
        self._make_current_projection_fit(image_payload)
        self.yplot.plot(image_payload.eproj, image_payload.energy)

    def _make_current_projection_fit(self, image_payload):
        try:
            if image_payload.time_calibration and image_payload.bunch_charge:
                xvar = image_payload.time_calibrated
                yvar = image_payload.current
                popt = image_payload.gaussfit_current.to_popt()
                stdev = image_payload.stdev_bunch_length * 1e12
                sigma_gauss = popt[2] *1e12
                self.gauss_beam_length_string.setHtml(f"σ = {stdev:.2f} ps<br><i>σ</i><sub>Gauss.</sub> = {sigma_gauss:.2f} ps")

            else:
                xvar = image_payload.time
                yvar = image_payload.tproj
                popt = image_payload.gaussfit_time.to_popt()
                # Only state the bunch length if there is a sigma, otherwise it has no meaning anyway
                self.gauss_beam_length_string.setHtml(f"")
        except (ValueError, RuntimeError, TypeError):
            self.gauss_beam_length_string.setHtml(f"")
        else:
            tgaussfit = gauss(xvar, *popt)
            # We scale the axis elsewhere, so we plot in the original coordinate syste, not wiht time!!!
            self.xplot.plot(image_payload.time, yvar, name="Data")
            self.xplot.plot(image_payload.time, tgaussfit, name="Fit", pen=pg.mkPen("red", width=3))




class CalibrationWatcher(QObject):
    # This is necessary in case the calibrations change at all at the
    # screen.  E.g. energy changes, position changes, dispersion changes..  etc.

    # XXX: What if scale factor is 0?
    # axis_calibration_signal = pyqtSignal(AxisCalibration)

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
        # should probably do this in tickValues instead, but this is way easier.
        strings = super().tickStrings(values, scale, spacing)
        if self.negate:
            # Add zero actually converts -0.0 to +0.0 whilst not doing anything else.
            strings = [str(-float(s) + 0.0) for s in strings]
        return strings
