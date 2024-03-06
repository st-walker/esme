import sys
import logging

from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QRunnable, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QMainWindow

from esme.gui.ui import mainwindow
from esme.gui.common import (make_default_i1_lps_machine,
                             make_default_b2_lps_machine,
                             set_tds_calibration_by_region)


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def start_gui():
    # make pyqt threadsafe
    # QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)

    main_window = LPSMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())



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
        self.screen_name_signal.connect(self.ui.tds_panel.set_region_from_screen_name)
        self.screen_name_signal.connect(self.ui.screen_display_widget.set_screen)

        # Connect the emission of the TDS panel's TDS voltage calibrations to here.
        self.ui.tds_panel.voltage_calibration_signal.connect(self.update_tds_voltage_calibration)

        self.update_screen_combo_box()

        self.connect_buttons()
        # self.setup_indicators()

        self.jddd_camera_window_process = None

        self.timer = self.build_main_timer(period=100)

        # Get the TDS child panel to emit any TDS calibrations it may
        # have stored, that we we can propagate them from here to
        # wherever else they need to be.
        self.ui.tds_panel.emit_calibrations()


    def update_location(self):
        self.update_screen_combo_box()

    def update_screen_combo_box(self):
        self.ui.select_screen_combobox.clear()
        self.ui.select_screen_combobox.addItems(self.machine.screens)

    def emit_current_screen_name(self):
        self.screen_name_signal.emit(self.get_selected_screen_name())

    def get_selected_screen_name(self) -> str:
        return self.ui.select_screen_combobox.currentText()

    def open_jddd_screen_window(self):
        self.jddd_camera_window_process = QtCore.QProcess()
        screen = self.get_selected_screen_name()
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen}/"
        self.jddd_camera_window_process.start(command)
        self.jddd_camera_window_process.waitForStarted()
        self.jddd_camera_window_process.finished.connect(self.close_jddd_screen_window)

    def update_tds_voltage_calibration(self, voltage_calibration):
        set_tds_calibration_by_region(self, voltage_calibration)
        print("Updating!!")
        self.ui.screen_display_widget.propagate_tds_calibration_signal(voltage_calibration)

        # self.ui.screen_display_widget.tds_calibration_signal.emit(voltage_calibration)

    def get_non_streaking_xaxis(self):
        """Energy axis or just offset axis (if no dispersion)"""
        if self.i1_radio_button.isChecked():
            return self.xplot.getAxis("bottom")
        elif self.b2_radio_button.isChecked():
            return self.yplot.getAxis("right")
        else:
            raise RuntimeError("either I1 nor B2 selected")

    def get_streaking_xaxis(self):
        if self.i1_radio_button.isChecked():
            return self.yplot.getAxis("right")
        elif self.b2_radio_button.isChecked():
            return self.xplot.getAxis("bottom")
        else:
            raise RuntimeError("either I1 nor B2 selected")

    def send_to_logbook(self):
        send_widget_to_log(self, author="Longitudinal Diagnostics Utility")

    def setup_logger_tab(self):
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

    def set_i1(self):
        LOG.debug("Setting location to I1")
        self.machine = self.i1machine
        self.update_screen_combo_box()

    def set_b2(self):
        LOG.debug("Setting location to B2")
        self.machine = self.b2machine
        self.update_screen_combo_box()

    def connect_buttons(self):
        # Location buttons
        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)
        self.ui.jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.select_screen_combobox.activated.connect(self.select_screen)
        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)
        self.ui.action_close.triggered.connect(self.close)

    def select_screen(self, _):
        screen_name = self.ui.select_screen_combobox.currentText()
        self.screen_name_signal.emit(screen_name)
        self.machine.set_kickers_for_screen(screen_name)

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

    def closeEvent(self, event):
        self.close_jddd_screen_window()
        # Close screen display widget where we have threads running
        # here explicitly otherwise we have a problem.  Could also
        # emit a closing signal.  This is fine.
        self.ui.screen_display_widget.close()

    def close_jddd_screen_window(self):
        try:
            self.jddd_camera_window_process.close()
        except AttributeError:
            pass
        else:
            self.jddd_camera_window_process = None



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
    start_gui()
