# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
from importlib_resources import files

import logging
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.tds import Ui_tds_control_panel
from esme.gui.common import build_default_machine_interface

from .calibrator import CalibrationMainWindow

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class TDSControl(QtWidgets.QWidget):
    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        if machine is None:
            self.machine = build_default_machine_interface()
        else:
            self.machine = machine

        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        self.calibration_window = CalibrationMainWindow(self)
        
        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update)
        self.timer.start(1000)

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

    def update_volatge(self, voltage):
        amplitude = self.mapping.get_amplitude(voltage)
        self.ui.tds_amplitude_spinbox.setValue(amplitude)
