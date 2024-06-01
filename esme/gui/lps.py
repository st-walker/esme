import logging
import os
import sys

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

from esme.gui.widgets.common import (
    make_exception_hook,
    send_widget_to_log,
    set_tds_calibration_by_region,
)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def start_lps_gui() -> None:
    app_name = "Tedious"
    sys.excepthook = make_exception_hook(app_name)
    app = QApplication(sys.argv)
    app.setOrganizationName("lps-tools")
    app.setApplicationName(app_name)

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    main_window = LPSMainWindow()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


class LPSMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = mainwindow.Ui_tan1_mainwindow()
        self.ui.setupUi(self)

        # Connect signals for screen name which we select in this
        # widget and have to propagate to the child widgets.
        self.ui.area.screen_name_signal.connect(
            self.ui.tds_panel.set_region_from_screen_name
        )
        self.ui.area.screen_name_signal.connect(self.ui.imaging_widget.set_screen)
        self.ui.area.screen_name_signal.connect(self.ui.machine_state_widget.set_screen)
        self.ui.area.screen_name_signal.connect(
            self.ui.special_bunch_panel.set_kickers_for_screen
        )
        self.ui.area.screen_name_signal.connect(self.ui.imaging_widget.set_screen)

        # Connect the emission of the TDS panel's TDS voltage calibrations to here.
        self.ui.tds_panel.voltage_calibration_signal.connect(
            self.update_tds_voltage_calibration
        )

        self.connect_buttons()

        # Get the TDS child panel to emit any TDS calibrations it may
        # have stored, that we we can propagate them from here to
        # wherever else they need to be.
        self.ui.tds_panel.emit_calibrations()

        # Emit initial screen name to any widgets.
        self.ui.area.emit_current_screen_name()

    def update_tds_voltage_calibration(self, voltage_calibration) -> None:
        set_tds_calibration_by_region(self, voltage_calibration)
        self.ui.imaging_widget.set_tds_calibration(voltage_calibration)

    def send_to_logbook(self) -> None:
        send_widget_to_log(
            self, author="", title="Longitudinal Diagnostics Utility", severity="INFO"
        )

    def setup_logger_tab(self) -> None:
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def connect_buttons(self) -> None:
        # Menu buttons
        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)
        self.ui.action_close.triggered.connect(self.close)

    def closeEvent(self, event) -> None:
        self.ui.area.close()
        # Close screen display widget where we have threads running
        # here explicitly otherwise we have a problem.  Could also
        # emit a closing signal.  This is fine.
        self.ui.imaging_widget.close()


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record) -> None:
        msg = self.format(record)
        self.log_signal.emit(msg)


if __name__ == "__main__":
    start_lps_gui()
