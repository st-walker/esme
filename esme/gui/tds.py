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


import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.tds import Ui_tds_control_panel
from esme.gui.common import build_default_machine_interface, get_i1_calibration_config_dir
from esme.control.configs import load_calibration

from .calibrator import CalibrationMainWindow

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class TDSControl(QtWidgets.QWidget):
    calibration_signal = pyqtSignal(object)
    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        if machine is None:
            self.machine = build_default_machine_interface()
        else:
            self.machine = machine

        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        self.calibration_window = CalibrationMainWindow(self)

        self.calibration_window.calibration_signal.connect(self.apply_calibration)

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update)
        self.timer.start(250)

        # I have to load it like this so that the above signal is
        # emitted when the GUI event loop has started, if I just emit
        # it in the init naively then the signal receivers may not be
        # initialised at emit time, which means that the signal will
        # be missed completely.
        QTimer.singleShot(50, self.load_most_recent_calibration)


    def update_location(self, location):
        LOG.info(f"Setting location for TDSControl panel: {location=}")
        self.machine.set_measurement_location(location)

    def update(self):
        self.ui.tds_phase_readback_line.setText(f"{(self.machine.deflectors.get_phase_rb()):.2f}")
        self.ui.tds_amplitude_readback_line.setText(f"{self.machine.deflectors.get_amplitude_rb():.2f}")
        self.ui.tds_phase_spinbox.setValue(self.machine.deflectors.get_phase_sp())
        self.ui.tds_amplitude_spinbox.setValue(self.machine.deflectors.get_amplitude_sp())

    def connect_buttons(self):
        # TDS Buttons
        self.ui.tds_phase_spinbox.valueChanged.connect(self.machine.deflectors.set_phase)
        self.ui.tds_amplitude_spinbox.valueChanged.connect(self.machine.deflectors.set_amplitude)
        self.ui.tds_calibration_pushbutton.clicked.connect(self.calibration_window.show)

    def apply_calibration(self, mapping):
        self.mapping = mapping
        self.tds_voltage_spinbox.setEnabled(True)
        self.tds_voltage_spinbox.valueChanged.connect(self.update_voltage)
        self.calibration_signal.emit(mapping)

    def update_volatge(self, voltage):
        amplitude = self.mapping.get_amplitude(voltage)
        self.ui.tds_amplitude_spinbox.setValue(amplitude)

    def load_calibration_file(self, calibration_filename):
        calibration = load_calibration(calibration_filename)
        self.set_calibration_info_strings(calibration_filename)
        self.calibration_signal.emit(calibration)
        self.machine.deflectors.active_tds().calibration = calibration

    def set_calibration_info_strings(self, calibration_filename):
        short_filename = re.sub(f"^{Path.home()}", "~", str(calibration_filename))
        self.ui.calibration_file_path_label.setText(str(short_filename))

        time = datetime.fromtimestamp(os.path.getmtime(calibration_filename))
        tz = pytz.timezone("Europe/Berlin")
        local_hamburg_time = tz.localize(time)
        timestamp = local_hamburg_time.strftime(f"%Y-%m-%d-%H:%M:%S")
        self.ui.calibration_time_label_2.setText(timestamp)

    def load_most_recent_calibration(self):
        # TODO: Hardcoded for I1 only currently, what about BC2? Fix this!
        cdir = get_i1_calibration_config_dir()
        files = cdir.glob("*.toml")
        try:
            newest_file = next(iter(sorted(files, key=os.path.getmtime)))
        except StopIteration:
            msg = (f"No TDS calibration files found in {cdir}."
                    "  Make one by calibrating the TDS with the calibration tool")
            raise_message_box(text="No TDS Calibration found",
                              informative_text=msg,
                              title="Warning",
                              icon="Warning")
        else:
            self.load_calibration_file(newest_file)
