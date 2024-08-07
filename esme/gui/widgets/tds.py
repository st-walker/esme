import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from importlib.resources import files
from pathlib import Path

import pytz
from PyQt5 import QtWidgets
from PyQt5.QtCore import QSettings, QTimer, pyqtSignal

from esme import DiagnosticRegion
from esme.control.machines import LPSMachine
from esme.core import region_from_screen_name
from esme.gui.tds_calibrator import CalibrationMainWindow
from esme.gui.ui.tds import Ui_tds_control_panel
from esme.load import load_amplitude_voltage_mapping
from esme.calibration import AmplitudeVoltageMapping
from datetime import datetime

from .common import (
    get_machine_manager_factory,
    get_tds_calibration_config_dir,
    set_machine_by_region,
)

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

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


# The TDSControl loads any most recent calibrations and emits them.



@dataclass
class TaggedVoltageCalibration:
    avmapping: AmplitudeVoltageMapping
    filepath: Path


class TDSControl(QtWidgets.QWidget):
    voltage_calibration_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        (
            self.i1machine,
            self.b2machine,
        ) = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = (
            self.i1machine
        )  # Set initial machine choice to be for I1 diagnostics

        # Make the daughter Calibrator window
        self.calibration_window = CalibrationMainWindow(self)
        self.calibration_window.avmapping_signal.connect(self.update_tds_calibration)

        self.i1calibration, self.b2calibration = self.load_most_recent_calibrations()

        # Connect calibration window.
        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(2000)

        self._init_state_from_settings()

        self.emit_calibrations()

    def _init_state_from_settings(self) -> None:
        s = QSettings()
        if zc := s.value("i1/tds/zero-crossing") is not None:
            self.i1machine.deflector = zc
        if zc := s.value("b2/tds/zero-crossing") is not None:
            self.b2machine.deflector = zc

    def emit_calibrations(self) -> None:
        i1c = self.i1machine.deflector.calibration
        b2c = self.b2machine.deflector.calibration
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
        set_machine_by_region(self, region)  # Update self.machine.
        self.ui.go_to_zero_crossing_button.setEnabled(
            self.machine.deflector.zero_crossing is not None
        )

    def update_ui(self) -> None:
        """Update the various readback values and so on."""
        self.set_calibration_info_strings()
        self.ui.tds_phase_readback.setText(
            f"{(self.machine.deflector.get_phase_rb()):.2f}"
        )
        self.ui.tds_amplitude_readback.setText(
            f"{self.machine.deflector.get_amplitude_rb():.2f}"
        )
        self.ui.tds_phase_spinbox.setValue(self.machine.deflector.get_phase_sp())
        self.ui.tds_amplitude_spinbox.setValue(
            self.machine.deflector.get_amplitude_sp()
        )
        if self.machine.deflector.calibration:
            # We multiply by 1e-6 convert from V to MV.
            self.ui.tds_voltage_readback.setText(f"{(self.machine.deflector.get_voltage_rb() * 1e-6):.2f}")
            self.ui.tds_voltage_spinbox.setValue(self.machine.deflector.get_voltage_sp() * 1e-6)

    def connect_buttons(self) -> None:
        self.ui.tds_phase_spinbox.valueChanged.connect(self.set_phase)
        self.ui.tds_amplitude_spinbox.valueChanged.connect(self.set_amplitude)
        self.ui.tds_calibration_pushbutton.clicked.connect(self.calibration_window.show)
        self.ui.set_zero_crossing_button.clicked.connect(self.set_zero_crossing)
        self.ui.go_to_zero_crossing_button.clicked.connect(self.go_to_zero_crossing)
        self.ui.subtract_180deg_button.clicked.connect(self.subtract_180_deg)
        self.ui.add_180_deg_button.clicked.connect(self.add_180_deg)

    def set_voltage(self, voltage: float) -> None:
        self.machine.deflector.set_voltage(voltage * 1e6)
        

    def subtract_180_deg(self) -> None:
        phase = self.machine.deflector.get_phase_sp()
        self.machine.deflector.set_phase(phase - 180)

    def add_180_deg(self) -> None:
        phase = self.machine.deflector.get_phase_sp()
        self.machine.deflector.set_phase(phase + 180)

    def set_zero_crossing(self) -> None:
        self.machine.deflector.set_zero_crossing()
        self.ui.go_to_zero_crossing_button.setEnabled(True)
        zc = self.machine.deflector.zero_crossing
        self.ui.zero_crossing_label.setText(f"Zero Crossing: {zc:.4g}°")

    def go_to_zero_crossing(self) -> None:
        self.machine.deflector.set_phase_to_zero_crossing()

    def set_phase(self, phase: float) -> None:
        """Simply forward to active machine instance, could either be for I1 or B2"""
        self.machine.deflector.set_phase(phase)

    def set_amplitude(self, amplitude: float) -> None:
        """Simply forward to active machine instance, could either be for I1 or B2"""
        self.machine.deflector.set_amplitude(amplitude)

    def update_tds_calibration(self, calibration):
        if calibration.region is DiagnosticRegion.I1:
            self.b2machine.deflector.calibration = calibration
        elif calibration.region is DiagnosticRegion.B2:
            self.b2machine.deflector.calibration = calibration
        self.voltage_calibration_signal.emit(calibration)

    def set_calibration_info_strings(self) -> None:
        if self.i1machine is self.machine:
            tagged_calibration = self.i1calibration
        elif self.b2machine is self.machine:
            tagged_calibration = self.b2calibration
        
        if tagged_calibration is None:
            self.ui.calibration_file_path_label.setText("Missing Calibration")
            self.ui.calibration_time_label_2.setText("")
            return

        calibration_filename = Path(tagged_calibration.filepath)
        datetime = tagged_calibration.avmapping.calibration_time
        short_filename = os.sep.join(calibration_filename.parts[-1:])

        self.ui.calibration_file_path_label.setText(short_filename)
        self.ui.calibration_file_path_label.setToolTip(str(calibration_filename))

        tz = pytz.timezone("Europe/Berlin")
        local_hamburg_time = tz.localize(datetime)
        timestamp = local_hamburg_time.strftime(f"%Y-%m-%d-%H:%M:%S")
        self.ui.calibration_time_label_2.setText(timestamp)

    def set_missing_calibration_info_string(self):
        self.ui.calibration_file_path_label.setText("No TDS Calibration")
        self.ui.calibration_time_label_2.setText("")

    def load_most_recent_calibrations(self) -> dict[DiagnosticRegion, LPSMachine]:
        try:
            i1tagged_calib = load_most_recent_calibration(DiagnosticRegion.I1)
        except (StopIteration, FileNotFoundError):
            i1tagged_calib = None
        else:
            self.set_avmapping(i1tagged_calib.avmapping)

        try:
            b2tagged_calib = load_most_recent_calibration(DiagnosticRegion.B2)
        except (StopIteration, FileNotFoundError):
            b2tagged_calib = None
        else:
            self.set_avmapping(b2tagged_calib.avmapping)

        return i1tagged_calib, b2tagged_calib
    
    def set_avmapping(self, avmapping: AmplitudeVoltageMapping) -> None:
        area = avmapping.region
        if area is DiagnosticRegion.I1:
            self.i1machine.deflector.calibration = avmapping
        elif area is DiagnosticRegion.B2:
            self.b2machine.deflector.calibration = avmapping


def load_most_recent_calibration(section: DiagnosticRegion) -> TaggedVoltageCalibration:
    cdir = get_tds_calibration_config_dir() / DiagnosticRegion(section).name.lower()

    dirs = cdir.glob("*")
    for directory in iter(sorted(dirs, key=os.path.getmtime)):
        if not directory.is_dir():
            continue
        calibration_file = directory / "calibration.toml"
        
        try:
            avmapping = load_amplitude_voltage_mapping(calibration_file)
        except FileNotFoundError:
            pass
        else:
            LOG.info(f"Loading calibration file {calibration_file} for {section.name} TDS")
            return TaggedVoltageCalibration(avmapping, calibration_file)

    msg = f"Failed to find calibration file for {section.name} TDS in {cdir}"
    LOG.warning(msg)
    raise FileNotFoundError(msg)