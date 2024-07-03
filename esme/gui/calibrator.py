from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHeaderView,
    QStyledItemDelegate,
    QTableView,
)


@dataclass
class CalibrationContext:
    amplitude: float
    screen_name: str  # Drop down menu?  Automatically calculated from
    # machine when generating new row or whatever.  Unless loading?
    # then we don't want to look it up.  whenever we update snapshot,
    # we update this.  the question then is only if/when we update the
    # snapshot.
    snapshot: pd.Series
    beam_energy: float
    frequency: float
    # button to add new row?

    # stacked widgets: one table for i1, one for b2.
    # No need to show screen, user can just see it elsewhere.

    # Add new row = instantiates above with current values from machine.
    # when new screen is selected, r12_streaking updates.
    # initial screen name is just same as previous row's.S
    # but with phases blank.

    def r12_streaking(self) -> float:
        pass


@dataclass
class PhaseScan:
    """Phase scan. This is output from the calibration routine."""

    prange0: tuple[float, float]
    samples: int
    coms: npt.NDArray | None


@dataclass
class CalibrationSetpoint:
    """This is the combination of the input with its corresponding output."""

    context: CalibrationContext
    pscan0: PhaseScan
    pscan1: PhaseScan

    def c1(self):
        pass

    def c2(self):
        pass

    def v1(self):
        pass

    def v2(self):
        pass

    def vmean(self):
        pass


@dataclass
class TDSCalibration:
    setpoints: list[CalibrationSetpoint]


@dataclass
class CalibrationContext:
    amplitude: float
    screen_name: str
    snapshot: pd.Series
    beam_energy: float
    frequency: float

    def r12_streaking(self) -> float:
        return self.amplitude * self.frequency  # Example implementation


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
            "Screen",
            "𝘙₃₄ / mrad⁻¹",
            "𝐶₀ / µmps⁻¹",
            "𝐶₁ / µmps⁻¹",
            "𝘝₀ / MV",
            "𝘝₁ / MV",
            "𝘝 / MV",
        ]

    def rowCount(self, parent=None) -> int:
        return len(self.calibration.setpoints)

    def columnCount(self, parent=None) -> int:
        return len(self.headers)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None

        setpoint = self.calibration.setpoints[index.row()]

        if index.column() == 0:
            return setpoint.context.amplitude
        elif index.column() == 1:
            return setpoint.pscan0.prange0[0]
        elif index.column() == 2:
            return setpoint.pscan0.prange0[1]
        elif index.column() == 3:
            return setpoint.pscan1.prange0[0]
        elif index.column() == 4:
            return setpoint.pscan1.prange0[1]
        elif index.column() == 5:
            return setpoint.context.screen_name
        elif index.column() == 6:
            return setpoint.context.r12_streaking()
        elif index.column() == 7:
            return setpoint.c1()
        elif index.column() == 8:
            return setpoint.c2()
        elif index.column() == 9:
            return setpoint.v1()
        elif index.column() == 10:
            return setpoint.v2()
        elif index.column() == 11:
            return setpoint.vmean()

        return None

    def setData(self, index, value, role=Qt.EditRole):
        if not index.isValid() or role != Qt.EditRole:
            return False

        setpoint = self.calibration.setpoints[index.row()]

        try:
            if index.column() == 0:
                setpoint.context.amplitude = float(value)
            elif index.column() == 1:
                setpoint.pscan0.prange0 = (float(value), setpoint.pscan0.prange0[1])
            elif index.column() == 2:
                setpoint.pscan0.prange0 = (setpoint.pscan0.prange0[0], float(value))
            elif index.column() == 3:
                setpoint.pscan1.prange0 = (float(value), setpoint.pscan1.prange0[1])
            elif index.column() == 4:
                setpoint.pscan1.prange0 = (setpoint.pscan1.prange0[0], float(value))
            elif index.column() == 5:
                setpoint.context.screen_name = value
            else:
                return False
            self.dataChanged.emit(index, index, [Qt.DisplayRole])
            return True
        except ValueError:
            return False

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled

        if 0 <= index.column() <= 5:
            return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable

        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            return self.headers[section]
        else:
            return section + 1


class ScreenDelegate(QStyledItemDelegate):
    def __init__(self, screens, parent=None):
        super(ScreenDelegate, self).__init__(parent)
        self.screens = screens

    def createEditor(self, parent, option, index):
        combo = QComboBox(parent)
        combo.addItems(self.screens)
        return combo

    def setEditorData(self, editor, index):
        value = index.model().data(index, Qt.DisplayRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), Qt.EditRole)


def create_sample_data():
    context1 = CalibrationContext(10.0, "Screen1", pd.Series([1, 2, 3]), 5.0, 2.0)
    context2 = CalibrationContext(20.0, "Screen2", pd.Series([4, 5, 6]), 6.0, 3.0)
    pscan0 = PhaseScan((0.1, 0.2), 100, np.array([1, 2, 3]))
    pscan1 = PhaseScan((0.3, 0.4), 100, np.array([4, 5, 6]))
    setpoint1 = CalibrationSetpoint(context1, pscan0, pscan1)
    setpoint2 = CalibrationSetpoint(context2, pscan0, pscan1)
    return TDSCalibration([setpoint1, setpoint2])


if __name__ == "__main__":
    app = QApplication([])

    calibration_data = create_sample_data()
    model = CalibrationTableModel(calibration_data)

    view = QTableView()
    view.setModel(model)

    screens = ["Screen1", "Screen2", "Screen3"]
    delegate = ScreenDelegate(screens)
    view.setItemDelegateForColumn(5, delegate)

    # Set the last column to stretch and fill the remaining horizontal space
    header = view.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.Interactive)  # Default for all columns
    header.setSectionResizeMode(11, QHeaderView.Stretch)  # Stretch the last column

    view.show()
    app.exec()
