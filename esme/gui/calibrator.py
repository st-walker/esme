import sys
from pathlib import Path
import logging

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
from matplotlib import cm
import numpy as np

from esme.gui.ui import calibration
from esme.gui.common import build_default_machine_interface, get_i1_calibration_config_dir
from esme.calibration import TDSCalibrator
from esme.control.configs import load_calibration

def com_label(ax):
    ax.set_ylabel(r"$y_\mathrm{com}$")


def phase_label(ax):
    ax.set_xlabel(r"$\phi$ / deg")


class CalibrationMainWindow(QMainWindow):
    calibration_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = calibration.Ui_MainWindow()
        self.ui.setupUi(self)

        self.machine = build_default_machine_interface()

        self.ui.load_calib_button.clicked.connect(self.load_calibration_file)
        self.ui.apply_calib_button.clicked.connect(self.apply_calibration_button)

        self.calibration = None

        self.setup_plots()

    @property
    def tds(self):
        return self.machine.deflectors.active_tds()

    def setup_plots(self):
        ax1 = self.ui.phase_com_plot_widget.axes
        ax2 = self.ui.amp_voltage_plot_widget.axes
        com_label(ax1)
        phase_label(ax1)
        ax1.set_title("Center of Mass against TDS Phase")

        ax2.set_xlabel("Amplitude / %")
        ax2.set_ylabel("Voltage / MV")
        ax2.set_title("TDS Calibration")

        # ax3.set_ylabel("Streak $\mathrm{\mu{}m\,/\,ps}$")

    # def main_loop(self):
    #     for phi0, phi1, amp in zip(self.parse_input_box()):
    #         self.tds.set_amplitude(amp)

    #         self.tds.set_phase_sp(phi0)

    # def do_one_setpoint(self, phi0, phi1, amplitude):
    #     phis = np.linspace(phi0, phi1, num=10)

    def parse_input_box(self):
        phi0 = [-30, -20, 10, 5]
        phi1 = [10, 20, 30, 50]
        amps = [5, 10, 15, 20]

        assert len(phi0) == len(phi1)
        assert len(amps) == len(phi1)

        return phi0, phi1, amps

    def update_calibration_plots(self):
        voltages = self.calibration.get_voltages()
        amplitudes = self.calibration.get_amplitudes()
        ax = self.ui.amp_voltage_plot_widget.axes
        ax.clear()
        self.setup_plots()
        ax.scatter(amplitudes, voltages * 1e-6)

        mapping = self.calibration.get_calibration_mapping()
        sample_ampls, fit_voltages = mapping.get_voltage_fit_line()
        m, c = mapping.get_voltage_fit_parameters()

        # label = fr"Fit: $m={abs(m)*1e-6:.3f}\mathrm{{MV}}\,/\,\%$, $c={c*1e-6:.2f}\,\mathrm{{MV}}$"
        ax.plot(sample_ampls, fit_voltages*1e-6)
        # ax.legend()
        self.draw()

    def get_plot_widgets(self):
        return [self.ui.amp_voltage_plot_widget,
                self.ui.phase_com_plot_widget,
                self.ui.phase_com_plot_widget]

    def draw(self):
        for wi in self.get_plot_widgets():
            wi.draw()

    def load_calibration_file(self):
        options = QFileDialog.Options()
        outdir = get_i1_calibration_config_dir()
        outdir.mkdir(parents=True, exist_ok=True)

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Save TDS Calibration",
            directory=str(outdir),
            filter="toml files (*.toml)",
            initialFilter="toml files",
            options=options,
        )

        if not fname:
            return

        self.calibration = load_calibration(fname)
        self.update_calibration_plots()

    def apply_calibration_button(self):
        if self.calibration is not None:
            self.calibration_signal.emit(self.calibration)
        

class CalibrationWorker(QObject):
    def __init__(self, machine, screen_name):
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
        LOG.info(f"Setting screen name for Screen Worker thread: {screen_name}")
        self.screen_name = screen_name
    
        

def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = CalibrationMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


