import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QRunnable, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import numpy as np

from esme.gui.ui import mainwindow
from esme.gui.widgets.common import (make_default_i1_lps_machine,
                                     make_default_b2_lps_machine,
                                     set_tds_calibration_by_region)
from collections import defaultdict
from dataclasses import dataclass
from functools import partial


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

from functools import wraps
import cProfile
import io
import pstats


def start_lps_gui() -> None:
    app = QApplication(sys.argv)
    main_window = LPSMainWindow()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


# @dataclass
# class BackgroundData:
#     imsum: np.ndarray | float = 0
#     n: int = 0

#     def add(self, image : np.ndarray) -> None:
#         self.imsum += image
#         self.n += 1

#     def mean(self) -> np.ndarray:
#         return self.imsum / self.n


class BackgroundCache:
    MAX_BG = 10
    """Class for storing background data for subtraction.  Idea is to """
    def __init__(self):
        self._cache = defaultdict(partial(deque, maxlen=self.MAX_BG))

    def add_image(self, screen_name: str, image: np.ndarray) -> None:
        self._cache[screen_name].add(image)

    def clear_cache(self, screen_name: str) -> None:
        self._cache[screen_name] = BackgroundData()

    def get_background(self, screen_name: str) -> np.ndarray:
        self._cache[screen_name].mean()





    
class LPSMainWindow(QMainWindow):
    screen_name_signal = pyqtSignal(str)
    # image_mode_signal = pyqtSignal(ImageTakingMode)

    def __init__(self):
        super().__init__()
        self.ui = mainwindow.Ui_MainWindow()
        self.ui.setupUi(self)

        # self.setup_logger_tab()

        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

        # Connect signals for screen name which we select in this
        # widget and have to propagate to the child widgets.
        self.ui.area.screen_name_signal.connect(self.ui.tds_panel.set_region_from_screen_name)
        self.ui.area.screen_name_signal.connect(self.ui.screen_display_widget.set_screen)
        self.ui.area.screen_name_signal.connect(self.ui.machine_state_widget.set_screen)

        # Connect the emission of the TDS panel's TDS voltage calibrations to here.
        self.ui.tds_panel.voltage_calibration_signal.connect(self.update_tds_voltage_calibration)

        self.connect_buttons()
        # self.setup_indicators()

        self.jddd_camera_window_process = None

        self.timer = self.build_main_timer(period=500)

        # Get the TDS child panel to emit any TDS calibrations it may
        # have stored, that we we can propagate them from here to
        # wherever else they need to be.
        self.ui.tds_panel.emit_calibrations()
        
        # Emit initial screen name to any widgets.
        self.ui.area.emit_current_screen_name()

    def update_tds_voltage_calibration(self, voltage_calibration) -> None:
        set_tds_calibration_by_region(self, voltage_calibration)
        self.ui.screen_display_widget.propagate_tds_calibration_signal(voltage_calibration)

    def get_non_streaking_xaxis(self) -> pg.AxisItem:
        """Energy axis or just offset axis (if no dispersion)"""
        if self.i1_radio_button.isChecked():
            return self.xplot.getAxis("bottom")
        elif self.b2_radio_button.isChecked():
            return self.yplot.getAxis("right")
        else:
            raise RuntimeError("either I1 nor B2 selected")

    def get_streaking_xaxis(self) -> pg.AxisItem:
        if self.i1_radio_button.isChecked():
            return self.yplot.getAxis("right")
        elif self.b2_radio_button.isChecked():
            return self.xplot.getAxis("bottom")
        else:
            raise RuntimeError("either I1 nor B2 selected")

    def send_to_logbook(self) -> None:
        send_widget_to_log(self, author="Longitudinal Diagnostics Utility")

    def setup_logger_tab(self) -> None:
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def set_i1(self) -> None:
        LOG.debug("Setting location to I1")
        self.machine = self.i1machine

    def set_b2(self) -> None:
        LOG.debug("Setting location to B2")
        self.machine = self.b2machine

    def connect_buttons(self) -> None:
        # Location buttons
        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)
        self.ui.action_close.triggered.connect(self.close)

    def set_bunch_control_enabled(self, enabled: bool) -> None:
        self.ui.i1_radio_button.setEnabled(enabled)
        self.ui.b2_radio_button.setEnabled(enabled)
        self.ui.special_bunch_panel.set_bunch_control_enabled(enabled)

    def build_main_timer(self, period: float) -> QTimer:
        """Period in ms"""
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.timeout.connect(self.update)
        timer.start(period)
        return timer

    def closeEvent(self, event) -> None:
        self.ui.area.close()
        # Close screen display widget where we have threads running
        # here explicitly otherwise we have a problem.  Could also
        # emit a closing signal.  This is fine.
        self.ui.screen_display_widget.close(event)

    def accumulate_background(self) -> None:
        """Take background data and cache the result.  The images can
        be used for background subtraction in the main image."""
        # with SnapshotAccumulator
        return


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

class BackgroundTaker(QRunnable):
    def __init__(self, screen, nbg):
        super().__init__()
        self.screen = screen
        self.nbg = nbg

    def run(self):
        background_images


if __name__ == "__main__":
    start_lps_gui()
