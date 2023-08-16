from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox

from esme.gui.ui import calibration


class CalibrationMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = calibration.Ui_MainWindow()
        self.ui.setupUi(self)
