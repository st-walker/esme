# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
from importlib_resources import files


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.tds import Ui_tds_control_panel
from esme.control.configs import build_simple_machine_from_config

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"


class TDSControl(QtWidgets.QWidget):
    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        if machine is None:
            self.machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH)
        else:
            self.machine = machine

        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def update(self):
        self.ui.tds_phase_readback_line.setText(f"{(self.machine.deflectors.get_phase_rb()):.1f}")
        self.ui.tds_amplitude_readback_line.setText(f"{self.machine.deflectors.get_amplitude_rb():.1f}")

    def connect_buttons(self):
        # TDS Buttons
        self.ui.tds_phase_spinbox.valueChanged.connect(self.machine.deflectors.set_phase)
        self.ui.tds_amplitude_spinbox.valueChanged.connect(self.machine.deflectors.set_amplitude)
        self.ui.tds_calibration_pushbutton.clicked.connect(self.open_calibration_window)

    def open_calibration_window(self):
        self.calibration_window = CalibrationMainWindow(self)
        self.calibration_window.show()
