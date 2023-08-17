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
    def __init__(self):
        super().__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        self.setup_logger_tab()

        self.machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH)

        self.populate_ui_initial_values()

        self.connect_buttons()

        self.image_plot = setup_screen_display_widget(self.ui.screen_display_widget)
        self.screen_worker, self.screen_thread = self.setup_screen_worker()

        self.ui.start_stop_button.clicked.connect(self.start_stop_special_bunches)

        self.setup_indicators()

        self.timer = self.build_main_timer(period=1000)

    @pyqtSlot()
    def start_stop_special_bunches(self):
        checked = self.ui.start_stop_button.isChecked()
        self.set_bunch_control_enabled(not checked)
        if checked:
            self.machine.sbunches.start_diagnostic_bunch()
            self.ui.start_stop_button.setText("Stop")
        else:
            self.machine.sbunches.stop_diagnostic_bunch()
            self.ui.start_stop_button.setText("Start")

    def setup_logger_tab(self):
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def update_screen_combo_box(self):
        self.ui.select_screen_combobox.clear()
        for screen_name in self.machine.screens.active_region_screen_names():
            self.ui.select_screen_combobox.addItem(screen_name)

    def populate_ui_initial_values(self):
        LOG.debug("Reading in initial values for the UI")
        self.update_screen_combo_box()
        self.read_from_machine()

    def set_location(self):
        self.clear_image()
        if self.ui.i1_radio_button.isChecked():
            LOG.debug("Setting location to I1")
            self.machine.set_measurement_location(DiagnosticRegion("I1"))
            self.update_screen_combo_box()
        elif self.ui.b2_radio_button.isChecked():
            LOG.debug("Setting location to B2")
            self.machine.set_measurement_location(DiagnosticRegion("B2"))
            self.update_screen_combo_box()

    def connect_buttons(self):
        # Location buttons
        self.ui.i1_radio_button.toggled.connect(self.set_location)
        self.ui.b2_radio_button.toggled.connect(self.set_location)
        self.ui.select_screen_combobox.currentIndexChanged.connect(self.configure_kickers)
        self.ui.use_fast_kickers_checkbox.stateChanged.connect(self.set_use_fast_kickers)

        # Bunch control buttons
        self.ui.beamregion_spinbox.valueChanged.connect(self.machine.sbunches.set_beam_region)
        self.ui.bunch_spinbox.valueChanged.connect(self.machine.sbunches.set_bunch_number)
        self.ui.npulses_spinbox.valueChanged.connect(self.machine.sbunches.set_npulses)

    def set_use_fast_kickers(self):
        if self.ui.use_fast_kickers_checkbox.isChecked():
            screen_name = self.get_selected_screen_name()
            kicker_sps = self.machine.screens.get_fast_kicker_setpoints_for_screen(screen_name)
            kicker_names = [k.name for k in kicker_sps]
            LOG.info(f"Enabling fast kickers for {screen_name}: kickers: {kicker_names}")
            # Just use first one and assume they are the same (they should be
            # configured as such on the doocs server...)
            self.machine.sbunches.set_kicker(kicker_names[0].name)
        else:
            self.machine.sbunches.set_dont_use_fast_kickers()

    def open_calibration_window(self):
        self.calibration_window = CalibrationMainWindow(self)
        self.calibration_window.show()

    def setup_indicators(self):
        self.indicator_timer = QTimer()
        indicator = self.ui.indicator_panel.add_indicator("TDS")
        indicator = self.ui.indicator_panel.add_indicator("Screen")
        indicator = self.ui.indicator_panel.add_indicator("Kicker")

    def set_bunch_control_enabled(self, enabled):
        self.ui.beamregion_spinbox.setEnabled(enabled)
        self.ui.bunch_spinbox.setEnabled(enabled)
        # self.ui.go_to_last_bunch_in_br_pushbutton.setEnabled(enabled)
        # self.ui.go_to_last_laserpulse_pushbutton.setEnabled(enabled)
        self.ui.i1_radio_button.setEnabled(enabled)
        self.ui.b2_radio_button.setEnabled(enabled)
        self.ui.use_fast_kickers_checkbox.setEnabled(enabled)
        self.ui.select_screen_combobox.setEnabled(enabled)
        self.ui.npulses_spinbox.setEnabled(enabled)

    def build_main_timer(self, period):
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        tds = self.machine.deflectors
        timer.timeout.connect(self.read_from_machine)
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

    def configure_kickers(self):
        self.clear_image()
        LOG.info(f"Configuring kickers for screen: {self.get_selected_screen_name()}")
        self.machine.set_kicker_for_screen(self.get_selected_screen_name())
        self.screen_worker.screen_name = self.get_selected_screen_name()

    def get_selected_screen_names(self):
        return self.ui.select_screen_combobox.currentText()

    def post_beam_image(self, image):
        items = self.image_plot.items
        assert len(items) == 1
        image_item = items[0]
        LOG.debug("Posting beam image...")
        # image = self.machine.screens.get_image(self.get_selected_screen_name())
        image_item.setImage(image)

    def clear_image(self):
        self.image_plot.items[0].clear()

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

class BunchPatternWatcher(QObject):
    pass

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

# i22.laser1
# i1.laser3
# i1.laser2
# i1.laser1M
