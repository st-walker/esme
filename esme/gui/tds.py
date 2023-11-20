# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
from importlib_resources import files
import os
import time
from pathlib import Path
from datetime import datetime
import pytz
import re
from dataclasses import dataclass

import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.tds import Ui_tds_control_panel
from esme.gui.common import (make_default_i1_lps_machine,
                             make_default_b2_lps_machine,
                             get_tds_calibration_config_dir,
                             set_machine_by_region,
                             raise_message_box)
from esme.control.configs import load_calibration
from esme import DiagnosticRegion
from esme.calibration import TDSCalibration
from esme.control.machines import LPSMachine
from esme.core import region_from_screen_name

from .calibrator import CalibrationMainWindow

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

# The way this panel works is that it has two machine interfaces, one
# for doing stuff in I1 and one for doing stuff in B2, stored under
# self.i1machine and self.b2machine, respectively.  When you want to
# swap the location, you simply update self.machine with the new
# instance of choice.  The buttons will automatically work for this
# newly chosen TDS because they all call self.machine.set/get_...

# The only tricky thing here is the calibration.  There may or may not
# be a calibration file in the relevant directory, and if there is not
# one then the TDS instances will have their .calibration attributes set to None.

# If the "active" TDS (determined by the TDSControl.machine instance)
# has no calibration, then that's OK.




@dataclass
class CalibrationMetadata:
    filename: str
    datetime: datetime


class TDSControl(QtWidgets.QWidget):
    voltage_calibration_signal = pyqtSignal(object)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        # Build two machine interfaces, one for I1 diagnostics and the other for B2 diagnostics.
        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine # Set initial machine choice to be for I1 diagnostics

        # Make the daughter Calibrator window
        self.calibration_window = CalibrationMainWindow(self)
        self.calibration_window.calibration_filename_signal.connect(self.update_tds_calibration_from_filename)

        # Connect calibration window.
        self.connect_buttons()

        # Cache both calibrations with their metadata for display purposes
        self.metadata = self.load_most_recent_calibrations()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(250)

        self.emit_calibrations()

    def emit_calibrations(self) -> None:
        i1c = self.i1machine.deflector.calibration
        b2c = self.b2machine.deflector.calibration
        print("EMITTING!", i1c, b2c)
        if i1c is not None:
            self.voltage_calibration_signal.emit(i1c)
        if b2c is not None:
            self.voltage_calibration_signal.emit(b2c)

    def set_region_from_screen_name(self, screen_name: str):
        region = region_from_screen_name(screen_name)
        self.update_region(region)

    def update_region(self, region: DiagnosticRegion) -> None:
        # Changing between state for using I1 or B2 diagnostics
        LOG.info(f"Setting region for TDSControl panel: {region=}")
        set_machine_by_region(self, region) # Update self.machine.
        # self.set_active_calibration()

    def update_ui(self) -> None:
        """Update the various readback values and so on."""
        self.set_calibration_info_strings()
        self.ui.tds_phase_readback_line.setText(f"{(self.machine.deflector.get_phase_rb()):.2f}")
        self.ui.tds_amplitude_readback_line.setText(f"{self.machine.deflector.get_amplitude_rb():.2f}")
        self.ui.tds_phase_spinbox.setValue(self.machine.deflector.get_phase_sp())
        self.ui.tds_amplitude_spinbox.setValue(self.machine.deflector.get_amplitude_sp())

    def connect_buttons(self) -> None:
        self.ui.tds_phase_spinbox.valueChanged.connect(self.set_phase)
        self.ui.tds_amplitude_spinbox.valueChanged.connect(self.set_amplitude)
        self.ui.tds_calibration_pushbutton.clicked.connect(self.calibration_window.show)

    def set_phase(self, phase: float) -> None:
        """Simply forward to active machine instance, could either be for I1 or B2"""
        self.machine.deflector.set_phase(phase)

    def set_amplitude(self, amplitude: float) -> None:
        """Simply forward to active machine instance, could either be for I1 or B2"""
        self.machine.deflector.set_amplitude(amplitude)

    def update_tds_calibration_from_filename(self, calibration_filename):
        calibration = load_calibration(calibration_filename)        
        if calibration.region is DiagnosticRegion.I1:
            self.b2machine.deflector.calibration = calibration
        elif calibration.region is DiagnosticRegion.B2:
            self.b2machine.deflector.calibration = calibration
        self.voltage_calibration_signal.emit(calibration)

    def set_calibration_info_strings(self) -> None:
        if self.machine.deflector.calibration is None:
            self.set_missing_calibration_info_string()
            return
        metadata = self.metadata[self.machine.region]
        calibration_filename = metadata.filename
        datetime = metadata.datetime
        short_filename = re.sub(f"^{Path.home()}", "~", str(calibration_filename))
        self.ui.calibration_file_path_label.setText(str(short_filename))

        tz = pytz.timezone("Europe/Berlin")
        local_hamburg_time = tz.localize(datetime)
        timestamp = local_hamburg_time.strftime(f"%Y-%m-%d-%H:%M:%S")
        self.ui.calibration_time_label_2.setText(timestamp)

    def set_missing_calibration_info_string(self):
        self.ui.calibration_file_path_label.setText("No TDS Calibration")        
        self.ui.calibration_time_label_2.setText("")

    def load_most_recent_calibrations(self) -> dict[DiagnosticRegion, LPSMachine]:
        try:
            self.i1machine.deflector.calibration, i1md = load_most_recent_calibration(DiagnosticRegion.I1)
        except StopIteration:
            self.i1.machine.deflector.calibration = None
            i1md = None

        try:
            self.b2machine.deflector.calibration, b2md = load_most_recent_calibration(DiagnosticRegion.B2)
        except StopIteration:
            self.b2machine.deflector.calibration = None
            b2md = None

        return {DiagnosticRegion.I1: i1md, DiagnosticRegion.B2: b2md}


def load_calibration_file(calibration_filename: str) -> tuple[TDSCalibration, CalibrationMetadata]:
    calibration = load_calibration(calibration_filename)
    file_birthday = datetime.fromtimestamp(os.path.getmtime(calibration_filename))
    return calibration, CalibrationMetadata(calibration_filename, file_birthday)

def load_most_recent_calibration(section: DiagnosticRegion) -> CalibrationMetadata:
    cdir = get_tds_calibration_config_dir() / DiagnosticRegion(section).name.lower()

    files = cdir.glob("*.toml")

    newest_file = next(iter(sorted(files, key=os.path.getmtime)))
    return load_calibration_file(newest_file)
