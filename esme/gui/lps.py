import sys
import logging
from collections import defaultdict
from functools import partial

from PyQt5.QtCore import QObject, QRunnable, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph as pg
import numpy as np

from esme.gui.ui import mainwindow
from esme.gui.widgets.common import (get_machine_manager_factory,
                                    set_tds_calibration_by_region,
                                     send_widget_to_log)


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

def start_lps_gui() -> None:
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

        # Connect signals for screen name which we select in this
        # widget and have to propagate to the child widgets.
        self.ui.area.screen_name_signal.connect(self.ui.tds_panel.set_region_from_screen_name)
        self.ui.area.screen_name_signal.connect(self.ui.screen_display_widget.set_screen)
        self.ui.area.screen_name_signal.connect(self.ui.machine_state_widget.set_screen)
        self.ui.area.screen_name_signal.connect(self.ui.special_bunch_panel.set_kickers_for_screen)

        # Connect the emission of the TDS panel's TDS voltage calibrations to here.
        self.ui.tds_panel.voltage_calibration_signal.connect(self.update_tds_voltage_calibration)

        self.connect_buttons()

        # Get the TDS child panel to emit any TDS calibrations it may
        # have stored, that we we can propagate them from here to
        # wherever else they need to be.
        self.ui.tds_panel.emit_calibrations()
        
        # Emit initial screen name to any widgets.
        self.ui.area.emit_current_screen_name()

    def update_tds_voltage_calibration(self, voltage_calibration) -> None:
        set_tds_calibration_by_region(self, voltage_calibration)
        self.ui.screen_display_widget.propagate_tds_calibration_signal(voltage_calibration)

    def send_to_logbook(self) -> None:
        send_widget_to_log(self, author="", title="Longitudinal Diagnostics Utility", severity="INFO")

    def setup_logger_tab(self) -> None:
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def connect_buttons(self) -> None:
        # Menu buttons
        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)
        self.ui.action_close.triggered.connect(self.close)

        sdw = self.ui.screen_display_widget
        # Measurements group box buttons.
        self.ui.write_to_logbook_button.clicked.connect(sdw.save_to_elog_signal.emit)
        self.ui.take_background_button.clicked.connect(sdw.take_background_signal.emit)
        self.ui.subtract_bg_checkbox.stateChanged.connect(sdw.subtract_background_signal.emit)
        # self.ui.current_profile_button.clicked.connect(sdw.current_profile_button)
        

    def closeEvent(self, event) -> None:
        self.ui.area.close()
        # Close screen display widget where we have threads running
        # here explicitly otherwise we have a problem.  Could also
        # emit a closing signal.  This is fine.
        self.ui.screen_display_widget.close()


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


if __name__ == "__main__":
    start_lps_gui()
