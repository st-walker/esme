import logging
import os
import sys

import yaml
from PyQt5.QtCore import QObject, QProcess, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

from esme.control.dint import DOOCSInterface
from esme.core import DiagnosticRegion
from esme.gui.ui import mainwindow
from esme.gui.widgets.common import (
    DEFAULT_CONFIG_PATH,
    get_machine_manager_factory,
    send_widget_to_log,
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
    "blms_and_toroids": "-file XFEL_BLM_TOROID_alarm_overview.xml",
}
_OPEN_IMAGE_ANALYSIS_CONFIG_LINE = (
    "cd /home/xfeloper/released_software/ImageAnalysisConfigurator"
    " ; bash /home/xfeloper/released_software/ImageAnalysisConfigurator"
    "/start_imageanalysis_configurator"
)


def start_lps_gui() -> None:
    app_name = "TDSChum"
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
        self._init_blms()
        self._init_target_control()

        self.di = DOOCSInterface()

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
        self.ui.area.region_signal.connect(self.ui.blm_stack.set_widget_by_region)
        self.ui.area.region_signal.connect(self.ui.target_stack.set_widget_by_region)

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

        self.timer = QTimer()
        self.timer.timeout.connect(self._set_beam_on_off_text)
        self.timer.start(1000)

    def _init_target_control(self) -> None:
        f = get_machine_manager_factory()
        i1d_target_def = f.make_target_definitions(DiagnosticRegion.I1)
        b2d_target_def = f.make_target_definitions(DiagnosticRegion.B2)
        self.ui.target_stack.set_target_widget(DiagnosticRegion.I1, i1d_target_def)
        self.ui.target_stack.set_target_widget(DiagnosticRegion.B2, b2d_target_def)

    def _init_blms(self) -> None:
        with open(DEFAULT_CONFIG_PATH, "r") as f:
            blm_definitions = yaml.full_load(f)["blms"]
            for region_name, blm_names in blm_definitions.items():
                self.ui.blm_stack.add_blms(DiagnosticRegion[region_name], blm_names)

    def update_tds_voltage_calibration(self, voltage_calibration) -> None:
        self.ui.imaging_widget.set_tds_calibration(voltage_calibration)

    def send_to_logbook(self) -> None:
        send_widget_to_log(self, author="xfeloper")

    def setup_logger_tab(self) -> None:
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def toggle_beam_on_off(self):
        if self.di.get_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"):
            self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)
        else:
            self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def _set_beam_on_off_text(self):
        if not self.di.get_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"):
            button_text = "Beam On"
        else:
            button_text = "Beam Off"
        self.ui.beam_on_off_button.setText(button_text)

    def connect_buttons(self) -> None:
        self.ui.beam_on_off_button.clicked.connect(self.toggle_beam_on_off)
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

        self.ui.action_image_analysis_server.triggered.connect(
            self._open_image_analysis_server
        )

    def _open_image_analysis_server(self):
        qproc = QProcess()
        qproc.setWorkingDirectory(
            "/home/xfeloper/released_software/ImageAnalysisConfigurator"
        )
        qproc.start(
            "bash",
            [
                "/home/xfeloper/released_software/ImageAnalysisConfigurator/start_imageanalysis_configurator"
            ],
        )
        # qproc.waitForStarted()
        self._processes["image_analysis_server"] = qproc

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
