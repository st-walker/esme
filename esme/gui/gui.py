import sys

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox


from esme.gui.ui import mainwindow
from esme.gui.configs import build_simple_machine_from_config

DEFAULT_CONFIG_PATH = "/Users/xfeloper/user/stwalker/esme/esme/gui/defaultconf.yml"

def start_gui():
    # make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)
    # path = os.path.join(
    #     os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png'
    # )
    # app.setWindowIcon(QtGui.QIcon(path))

    main_window = LPSMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())

class LPSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)
        # log_handler = QPlainTextEditLogger()
        # logging.getLogger().addHandler(log_handler)
        # log_handler.log_signal.connect(self.ui.log_output_widget.append)

        self.machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH)

        self.timer = QTimer()
        # self.timer.timeout.connect(self.post_image_from_screen)
        self.timer.start(100)

        for screen_name in self.machine.kickerop.screen_names:
            self.ui.comboBox.addItem(screen_name)

        self.ui.comboBox.currentIndexChanged.connect(self.configure_kicker)

    def configure_kicker(self):
        self.machine.kickerop.configure_kicker_for_screen(self.ui.comboBox.currentText())
            

if __name__ == "__main__":
    start_gui()
