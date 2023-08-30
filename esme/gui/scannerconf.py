from esme.gui.ui import scanner_config

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    conf = ScannerConfDialog()
    conf.show()
    sys.exit(app.exec_())
