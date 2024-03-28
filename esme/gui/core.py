import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit
from PyQt5.QtCore import Qt

class ReadOnlyNumberDisplay(QLineEdit):
    def __init__(self, parent=None, number=0.0, precision=2# , parent=None
                 ):
        super().__init__(parent=parent)
        self.precision = precision
        self.initUI(number)

    def initUI(self, number):
        # Set the widget to read-only to make it clear it's not editable
        self.setReadOnly(True)

        # Enable text copying
        self.setContextMenuPolicy(Qt.ActionsContextMenu)

        # Set the precision and display the number
        self.setNumber(number)
        
        # # Apply some styling to indicate it's read-only (e.g., a lighter background color)
        self.setStyleSheet("background-color: #f0f0f0; color: #333; border: 1px solid #ccc;")
        # # Layout
        # layout = QVBoxLayout()
        # layout.addWidget(self.lineEdit)
        # self.setLayout(layout)

    def setNumber(self, number):
        # Format the number with the given precision and display it
        self.setText(f"{number:.{self.precision}f}")
