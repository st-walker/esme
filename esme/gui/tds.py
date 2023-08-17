# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.tds import Ui_tds_control_panel


class TDSControl(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.ui = Ui_tds_control_panel()
        self.ui.setupUi(self)

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)

    def update(self):
        self.ui.tds_phase_readback_line.setText(str(self.machine.deflectors.get_phase_rb()))
        self.ui.tds_amplitude_readback_line.setText(str(self.machine.deflectors.get_amplitude_rb()))

    def connect_buttons(self):
        # TDS Buttons
        self.ui.tds_phase_spinbox.valueChanged.connect(self.machine.deflectors.set_phase)
        self.ui.tds_amplitude_spinbox.valueChanged.connect(self.machine.deflectors.set_amplitude)
        self.ui.tds_calibration_pushbutton.clicked.connect(self.open_calibration_window)
