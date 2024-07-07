from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QWidget


@dataclass
class PhaseScan:
    """Phase scan. This is output from the calibration routine."""

    prange: Optional[Tuple[float, float]] = None
    images: npt.NDArray = field(default_factory=lambda: np.array([]))

    def phases(self):
        pass

    def coms(self):
        return None

    def cal(self):
        return None


@dataclass
class CalibrationContext:
    amplitude: float = 0.0
    screen_name: str = ""
    snapshot: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    beam_energy: float = 0.0
    frequency: float = 0.0

    def r12_streaking(self) -> float:
        return self.amplitude * self.frequency  # Example implementation


@dataclass
class CalibrationSetpoint:
    """This is the combination of the input with its corresponding output."""

    amplitude: float = 0.0
    pscan0: PhaseScan = field(default_factory=PhaseScan)
    pscan1: PhaseScan = field(default_factory=PhaseScan)


@dataclass
class TDSCalibration:
    context: CalibrationContext = field(default_factory=CalibrationContext)
    setpoints: List[CalibrationSetpoint] = field(
        default_factory=lambda: [CalibrationSetpoint() for _ in range(5)]
    )

    def v0(self) -> List[float]:
        pass

    def v1(self) -> List[float]:
        pass

    def vmean(self) -> float:
        pass


class CalibrationTableModel(QAbstractTableModel):
    def __init__(self, calibration: TDSCalibration):
        super().__init__()
        self.calibration = calibration
        self.headers = [
            "Amplitude / %",
            "  𝜙₀₀ / °  ",
            "  𝜙₀₁ / °  ",
            "  𝜙₁₀ / °  ",
            "  𝜙₁₁ / °  ",
            "𝐶₀ / µmps⁻¹",
            "𝐶₁ / µmps⁻¹",
            "𝘝₀ / MV",
            "𝘝₁ / MV",
            "𝘝 / MV",
        ]

    def rowCount(self, parent=None):
        return len(self.calibration.setpoints)

    def columnCount(self, parent=None):
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        setpoint = self.calibration.setpoints[index.row()]

        if role == Qt.DisplayRole:
            if index.column() == 0:
                return setpoint.amplitude if setpoint.amplitude != 0.0 else ""
            elif index.column() == 1:
                return (
                    setpoint.pscan0.prange[0]
                    if setpoint.pscan0.prange and setpoint.pscan0.prange[0] != 0.0
                    else ""
                )
            elif index.column() == 2:
                return (
                    setpoint.pscan0.prange[1]
                    if setpoint.pscan0.prange and setpoint.pscan0.prange[1] != 0.0
                    else ""
                )
            elif index.column() == 3:
                return (
                    setpoint.pscan1.prange[0]
                    if setpoint.pscan1.prange and setpoint.pscan1.prange[0] != 0.0
                    else ""
                )
            elif index.column() == 4:
                return (
                    setpoint.pscan1.prange[1]
                    if setpoint.pscan1.prange and setpoint.pscan1.prange[1] != 0.0
                    else ""
                )
            elif index.column() == 5:
                return (
                    setpoint.pscan0.coms() if setpoint.pscan0.coms() is not None else ""
                )
            elif index.column() == 6:
                return (
                    setpoint.pscan1.coms() if setpoint.pscan1.coms() is not None else ""
                )
            elif index.column() == 7:
                return (
                    setpoint.pscan0.cal() if setpoint.pscan0.cal() is not None else ""
                )
            elif index.column() == 8:
                return (
                    setpoint.pscan1.cal() if setpoint.pscan1.cal() is not None else ""
                )
            elif index.column() == 9:
                return (
                    self.calibration.context.r12_streaking()
                    if self.calibration.context.r12_streaking() != 0.0
                    else ""
                )

        if role == Qt.BackgroundRole:
            if index.column() in {1, 2, 5, 7}:
                color = QColor(255, 182, 193)  # Light salmon color
                color.setAlphaF(0.1)  # Set alpha to 0.1
                return QBrush(color)
            elif index.column() in {3, 4, 6, 8}:
                color = QColor(173, 216, 230)  # Baby blue color
                color.setAlphaF(0.1)  # Set alpha to 0.1
                return QBrush(color)

        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        if index.row() >= len(self.calibration.setpoints):
            return False

        setpoint = self.calibration.setpoints[index.row()]

        try:
            value = float(value) if value != "" else None
            if index.column() == 0:
                setpoint.amplitude = value if value is not None else 0.0
            elif index.column() == 1:
                prange = (
                    (value, setpoint.pscan0.prange[1])
                    if value is not None
                    else (None, setpoint.pscan0.prange[1])
                )
                setpoint.pscan0.prange = prange
            elif index.column() == 2:
                prange = (
                    (setpoint.pscan0.prange[0], value)
                    if value is not None
                    else (setpoint.pscan0.prange[0], None)
                )
                setpoint.pscan0.prange = prange
            elif index.column() == 3:
                prange = (
                    (value, setpoint.pscan1.prange[1])
                    if value is not None
                    else (None, setpoint.pscan1.prange[1])
                )
                setpoint.pscan1.prange = prange
            elif index.column() == 4:
                prange = (
                    (setpoint.pscan1.prange[0], value)
                    if value is not None
                    else (setpoint.pscan1.prange[0], None)
                )
                setpoint.pscan1.prange = prange
            else:
                return False
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True
        except ValueError:
            return False

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        if index.column() in {0, 1, 2, 3, 4}:
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.headers[section]
        else:
            return section + 1


class CalibrationTableView(QTableView):
    # def __init__(self, model, parent=None):
    #     super(CustomTableView, self).__init__(parent)
    #     self.setModel(model)

    #     # Set the last column to stretch and fill the remaining horizontal space
    #     header = self.horizontalHeader()
    #     header.setSectionResizeMode(QHeaderView.Interactive)  # Default for all columns
    #     header.setSectionResizeMode(9, QHeaderView.Stretch)  # Stretch the last column

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace:
            index = self.currentIndex()
            if index.isValid():
                self.model().setData(index, "", Qt.EditRole)
        else:
            super().keyPressEvent(event)


def create_sample_data():
    return TDSCalibration()


if __name__ == "__main__":
    app = QApplication([])

    calibration_data = create_sample_data()
    model = CalibrationTableModel(calibration_data)

    custom_view = CustomTableView(model)

    window = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(custom_view)
    window.setLayout(layout)
    window.show()

    app.exec()
