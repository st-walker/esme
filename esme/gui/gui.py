import sys
import logging
import time
from importlib_resources import files

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
from matplotlib import cm
import numpy as np


from esme.gui.ui import mainwindow
from .calibration import CalibrationMainWindow
from esme.control.configs import build_simple_machine_from_config
from esme.control.sbunches import DiagnosticRegion
from esme.control.pattern import get_beam_regions, get_bunch_pattern

pg.setConfigOption("useNumba", True)
pg.setConfigOption("imageAxisOrder", "row-major")


DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"

# DEFAULT_CONFIG_PATH = "/Users/xfeloper/user/stwalker/esme/esme/gui/defaultconf.yml"
# DEFAULT_CONFIG_PATH = "/Users/stuartwalker/repos/esme/esme/gui/defaultconf.yml"

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def start_gui():
    # make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)

    main_window = LPSMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())

class LPSMainWindow(QMainWindow):
    location = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        self.setup_logger_tab()

        self.machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH)
        # Share same machine instance
        self.ui.special_bunch_panel.machine = self.machine
        self.ui.tds_panel.machine = self.machine

        self.connect_buttons()
        self.setup_indicators()

        self.image_plot = setup_screen_display_widget(self.ui.screen_display_widget)
        self.screen_worker, self.screen_thread = self.setup_screen_worker()

        self.timer = self.build_main_timer(period=1000)

    def setup_logger_tab(self):
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def set_i1(self):
        LOG.debug("Setting location to I1")
        self.machine.set_measurement_location(DiagnosticRegion("I1"))

    def set_b2(self):
        LOG.debug("Setting location to B2")
        self.machine.set_measurement_location(DiagnosticRegion("B2"))

    def connect_buttons(self):
        # Location buttons
        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)

    def setup_indicators(self):
        indicator = self.ui.indicator_panel.add_indicator("TDS")
        indicator = self.ui.indicator_panel.add_indicator("Screen")
        indicator = self.ui.indicator_panel.add_indicator("Kicker")

    def set_bunch_control_enabled(self, enabled):
        self.ui.i1_radio_button.setEnabled(enabled)
        self.ui.b2_radio_button.setEnabled(enabled)
        self.ui.special_bunch_panel.set_bunch_control_enabled(enabled)

    def build_main_timer(self, period):
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.timeout.connect(self.update)
        timer.start(period)
        return timer

    def setup_screen_worker(self):
        LOG.debug("Initialising screen worker thread")
        screen_worker = ScreenWatcher(self.machine)
        screen_thread = QThread()
        screen_worker.moveToThread(screen_thread)
        screen_thread.started.connect(screen_worker.run)
        screen_worker.image_signal.connect(self.post_beam_image)
        screen_thread.start()
        return screen_worker, screen_thread

    def post_beam_image(self, image):
        items = self.image_plot.items
        assert len(items) == 1
        image_item = items[0]
        LOG.debug("Posting beam image...")
        # image = self.machine.screens.get_image(self.get_selected_screen_name())
        image_item.setImage(image)

    def closeEvent(self, event):
        self.screen_worker.kill = True
        self.screen_thread.terminate()
        self.screen_thread.wait()
        # self.screen_thread.exit()


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class ScreenWatcher(QObject):
    image_signal = pyqtSignal(np.ndarray)
    def __init__(self, machine):
        super().__init__()
        self.machine = machine
        self.screen_name = "OTRC.55.I1"
        self.kill = False

    def get_image(self):
        image = self.machine.screens.get_image(self.screen_name)
        LOG.info("Reading image from: %s", self.screen_name)
        return image # .astype(np.float32)

    def run(self):
        while not self.kill:
            time.sleep(0.1)
            image = self.get_image()
            if image is None:
                continue
            else:
                self.image_signal.emit(image)

    def update_screen_name(self, screen_name):
        self.screen_name = screen_name

def setup_screen_display_widget(widget):
    image_plot = widget.addPlot()
    image_plot.clear()
    image_plot.hideAxis("left")
    image_plot.hideAxis("bottom")
    image = pg.ImageItem(autoDownsample=True, border="k")

    image_plot.addItem(image)

    colormap = cm.get_cmap("viridis")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)

    image.setLookupTable(lut)
    # print(lut.shape)

    return image_plot


if __name__ == "__main__":
    start_gui()
