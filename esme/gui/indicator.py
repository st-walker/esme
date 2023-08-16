from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QAbstractButton


from PyQt5.QtCore import QPointF
from PyQt5.QtGui import QColor, QPainter, QBrush
from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox


from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from enum import Enum, auto


class State(Enum):
    GOOD = auto()
    BAD = auto()
    INDETERMINATE = auto()

class IndicatorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout()
        # layout = QGridLayout()
        # https://stackoverflow.com/questions/74252940/how-to-automatically-adjust-the-elements-of-qgridlayout-to-fit-in-a-row-when-the
        self.setLayout(layout)

    def get_indicator(self, label):
        self.layout.get_item_position(label, QFormLayout.LabelRole)

    def label_text_to_widget_map(self):
        layout = self.layout()
        result = {}
        for i in range(layout.rowCount()):
            text = layout.itemAt(i, QFormLayout.LabelRole).widget().text()
            indicator = layout.itemAt(i, QFormLayout.FieldRole).widget()
            result[text] = indicator
        return result

    def add_indicator(self, label):
        indicator = Indicator(parent=self)
        self.layout().addRow(label, indicator)

    def set_bad(self, label):
        indicator_map = self.label_text_to_widget_map()
        indicator = indicator_map[label]
        indicator.set_bad()


class Indicator(QWidget):
    SCALED_SIZE: float = 1200.0
    IS_GOOD_BOOL: bool = True
    IS_BAD_BOOL: bool = False

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setMinimumSize(24, 24)
        self.state = State.GOOD

        # Nice Green
        self.colour_good = QColor(148, 255, 67)
        # Nice Red
        self.colour_bad = QColor(227, 49, 30)
        # self.setDis

    def resizeEvent(self, QResizeEvent):
        self.update()

    def set_good(self):
        self.state = State.GOOD

    def set_bad(self):
        self.state = State.BAD

    def set_indeterminate(self):
        self.state = State.INDETERMINATE

    def toggle_state(self):
        if self.state is State.GOOD:
            self.set_bad()
        elif self.state is State.BAD:
            self.set_good()
        self.update()


    def paintEvent(self, paintevent: QPaintEvent):
        realSize = min(self.width(), self.height())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self.width() / 2, self.height() / 2)
        painter.scale(realSize / self.SCALED_SIZE, realSize / self.SCALED_SIZE)

        if self.state is State.GOOD:
            brush = QBrush(self.colour_good)
        else:
            brush = QBrush(self.colour_bad)
        painter.setBrush(brush)
        painter.drawEllipse(QPointF(0, 0), 400, 400)
