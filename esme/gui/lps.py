import logging
import os
import sys

from PyQt5.QtCore import QObject, QProcess, Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

from esme.gui.ui import mainwindow
from esme.gui.widgets.common import (
    get_machine_manager_factory,
    send_widget_to_log,
    set_tds_calibration_by_region,
)

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

_JDDD_RUN_ARGS = {
    "camera_status": "-file XFEL_Camera_Overview.xml -address",
    "pattern_builder": "-file bunch_pattern_server_pattern_builder.xml -address",
    "main_timing": "-file XFEL_MainTiming.xml",
    "b2_llrf": "-file Main_TDS_LLRF_Operation.xml -address XFEL.RF/LLRF.CONTROLLER/LLTDSB2/",
    "i1_llrf": "-file Main_TDS_LLRF_Operation.xml -address XFEL.RF/LLRF.CONTROLLER/LLTDSI1/",
    "b2_sbm": "-file XFEL_B2_Diag_bunches.xml -address XFEL.RF//LLTDSI1/",
    "i1_sbm": "-file XFEL_I1_Diag_bunches.xml -address XFEL.RF//LLTDSI1/",
}
_OPEN_IMAGE_ANALYSIS_CONFIG_LINE = (
    "cd /home/xfeloper/released_software/ImageAnalysisConfigurator"
    " ; bash /home/xfeloper/released_software/ImageAnalysisConfigurator"
    "/start_imageanalysis_configurator"
)


def start_lps_gui() -> None:
    app_name = "TDSFriend"
    # this somehow causes big problems...
    # sys.excepthook = make_exception_hook(app_name)
    app = QApplication(sys.argv)
    app.setOrganizationName("lps-tools")
    app.setApplicationName(app_name)

    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    main_window = LPSMainWindow()
    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


class LPSMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.ui = mainwindow.Ui_tdsfriend_mainwindow()
        self.ui.setupUi(self)
        self._init_target_control()

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
        self._processes: dict[str, QProcess] = {}

        # Get the TDS child panel to emit any TDS calibrations it may
        # have stored, that we we can propagate them from here to
        # wherever else they need to be.
        self.ui.tds_panel.emit_calibrations()

        # Emit initial screen name to any widgets.
        self.ui.area.emit_current_screen_name()

    def _init_target_control(self) -> None:
        f = get_machine_manager_factory()
        i1d_dump, i1d_undo_dump = f.make_dump_sequences(DiagnosticRegion.I1)
        b2d_dump, b2d_undo_dump = f.make_dump_sequences(DiagnosticRegion.B2)

        self.ui.target_stack.add_target_widget(
            DiagnosticRegion.I1, "I1D", i1d_forward, i1d_backard
        )

        self.ui.target_stack.add_target_widget(
            DiagnosticRegion.B2, "B2D", b2d_forward, b2d_backard
        )

    def update_tds_voltage_calibration(self, voltage_calibration) -> None:
        set_tds_calibration_by_region(self, voltage_calibration)
        self.ui.imaging_widget.set_tds_calibration(voltage_calibration)

    def send_to_logbook(self) -> None:
        send_widget_to_log(self, author="xfeloper")

    def setup_logger_tab(self) -> None:
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def connect_buttons(self) -> None:
        # Menu buttons
        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)
        self.ui.action_close.triggered.connect(self.close)

        jddd = _JDDD_RUN_ARGS
        self.ui.actionSpecial_Bunch_Midlayer_b2.triggered.connect(
            lambda: self._run_jddd_process("b2_sbm")
        )
        self.ui.actionSpecial_Bunch_Midlayer_i1.triggered.connect(
            lambda: self._run_jddd_process("i1_sbm")
        )
        self.ui.actionLLRF_i1.triggered.connect(
            lambda: self._run_jddd_process("i1_llrf")
        )
        self.ui.actionLLRF_b2.triggered.connect(
            lambda: self._run_jddd_process("b2_llrf")
        )
        self.ui.action_pattern_builder.triggered.connect(
            lambda: self._run_jddd_process("pattern_builder")
        )
        self.ui.action_camera_status.triggered.connect(
            lambda: self._run_jddd_process("camera_status")
        )

    def closeEvent(self, event) -> None:
        self.ui.area.close()
        # Close screen display widget where we have threads running
        # here explicitly otherwise we have a problem.  Could also
        # emit a closing signal.  This is fine.
        self.ui.imaging_widget.close()

    def _run_jddd_process(self, name):
        process = QProcess()
        process.start(f"jddd-run {_JDDD_RUN_ARGS[name]}")
        process.waitForStarted()
        self._processes[name] = process


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record) -> None:
        msg = self.format(record)
        self.log_signal.emit(msg)


if __name__ == "__main__":
    start_lps_gui()
