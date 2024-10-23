from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QTimer

from esme.control.tds import TransverseDeflector
from esme.gui.ui.tdsmini import Ui_tdsmini_panel

from dataclasses import dataclass

class MiniTDS(QWidget):
    def __init__(self, tds: TransverseDeflector, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self.ui = Ui_tdsmini_panel()
        self.ui.setupUi(self)
        self.tds = tds


        self._update_ui()
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(1000)
        self.connect_buttons()

    def connect_buttons(self):
        self.ui.phase_spinner.valueChanged.connect(self.tds.set_phase)
        self.ui.amplitude_spinner.valueChanged.connect(self.tds.set_amplitude)
        self.ui.add_180_deg_button.clicked.connect(self.tds.increment_180_degrees)
        self.ui.sub_180_deg_button.clicked.connect(self.tds.decrement_180_degrees)

    def _update_ui(self) -> None:
        # Amplitude setpoint and RB:
        amplitude_rb = self.tds.get_amplitude_rb()
        self.ui.amplitude_rb.setText(f"{amplitude_rb:.2f}°")        
        self.ui.amplitude_spinner.setValue(self.tds.get_amplitude_sp())

        # Phase setpoint and RB:
        phase_rb = self.tds.get_phase_rb()
        self.ui.phase_rb.setText(f"{phase_rb:.2f}°")        
        self.ui.phase_spinner.setValue(self.tds.get_phase_sp())

    def add_180deg(self) -> None:
        self.tds.set_phase(self.tds.get_phase_sp() + 180)
