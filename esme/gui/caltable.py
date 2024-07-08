from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QTableView


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
