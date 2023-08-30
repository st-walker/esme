import sys
from pathlib import Path
import logging
import time

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
from matplotlib import cm
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit

from esme.gui.ui import calibration
from esme.gui.common import build_default_machine_interface, get_i1_calibration_config_dir
from esme.calibration import TDSCalibrator
from esme.control.configs import load_calibration
from esme.image import filter_image

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


    def parse_amplitude_input_box(self):
        return [5, 10, 15, 20]

    def update_calibration_plots(self):
        voltages = self.calibration.get_voltages()
        amplitudes = self.calibration.get_amplitudes()
        ax = self.ui.amp_voltage_plot_widget.axes
        ax.clear()
        self.setup_plots()
        ax.scatter(amplitudes, voltages * 1e-6)

        # mapping = self.calibration.get_calibration_mapping()
        # sample_ampls, fit_voltages = mapping.get_voltage_fit_line()
        # m, c = mapping.get_voltage_fit_parameters()

        # label = fr"Fit: $m={abs(m)*1e-6:.3f}\mathrm{{MV}}\,/\,\%$, $c={c*1e-6:.2f}\,\mathrm{{MV}}$"
        # ax.plot(sample_ampls, fit_voltages*1e-6)
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
    CUT = 275
    def __init__(self, machine, screen_name, amplitudes):
        super().__init__()
        self.machine = machine
        self.screen_name = screen_name
        self.amplitudes = amplitudes
        self.kill = False

    def get_image(self):
        image = self.machine.screens.get_image(self.screen_name)
        LOG.info("Reading image from: %s", self.screen_name)
        return image # .astype(np.float32)

    def calibrate(self):
        self.machine.deflectors.active_tds().set_amplitude(0.0)
        time.sleep(1.0)
        image = self.machine.screens.get_image_raw(self.screen_name)
        image = image[:self.CUT]
        com = ndi.center_of_mass(image)
        ycom = com[1]
        yzero_crossing = ycom

        slopes = []
        for amplitude in self.amplitudes:
            self.machine.deflectors.active_tds().set_amplitude(amplitude)
            m1, m2 = self.calibrate_once(yzero_crossing)
            slopes.append((np.mean(abs(m1[0])), np.mean(abs(m2[0]))))

    def calibrate_once(self, zero_crossing):
        phis = np.linspace(-190, 190, num=191)
        ycoms = []
        tds = self.machine.deflectors.active_tds()
        tds.set_phase(phis[0])
        time.sleep(4)
        for phi in phis:
            time.sleep(0.1)
            tds.set_phase(phi)
            image = self.machine.screens.get_image_raw(self.screen_name)
            image = image[:275]
            image = filter_image(image, 0)
            com = ndi.center_of_mass(image)
            ycom = com[1]
            ycoms.append(ycom)

        from IPython import embed; embed()
        # time_s = phi / (3e9 * 360)
        # ycoms_m = np.array(ycoms) * 13.7369 * 1e-6

        m1, m2 = get_zero_crossing_slopes(phis, ycoms, zero_crossing=zero_crossing)
        return m1, m1 #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX JUST USING M1 HERE!!!...
            
    def run(self):
        slopes = self.calibrate()
        from IPython import embed; embed()

    def update_screen_name(self, screen_name):
        LOG.info(f"Setting screen name for Screen Worker thread: {screen_name}")
        self.screen_name = screen_name




def smooth(phase, com, window):
    w = window
    ycoms_smoothed = np.convolve(com, np.ones(w), "valid") / w
    phases_smoothed = np.convolve(phase, np.ones(w), "valid") / w
    return phases_smoothed, ycoms_smoothed


def get_monotonic_intervals(phases, coms):
    # We want to find where the centres of mass are consistently rising and falling.
    deriv = np.diff(coms)

    rising_mask = deriv > 0
    falling_mask = deriv <= 0

    (indices,) = np.where(np.diff(rising_mask))
    indices += 1  # off by one otherwise.

    phase_turning_points = phases[indices]

    piecewise_monotonic_coms = np.split(coms, indices)
    piecewise_monotonic_com_phases = np.split(phases, indices)

    yield from zip(piecewise_monotonic_com_phases, piecewise_monotonic_coms)


def line(x, a0, a1):
    return a0 + a1 * x


def linear_fit(indep_var, dep_var, dep_var_err=None):
    popt, pcov = curve_fit(line, indep_var, dep_var)
    # popt, pcov = curve_fit(line, indep_var, dep_var, sigma=dep_var_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0]  # y-intercept with error
    a1 = popt[1], perr[1]  # gradient with error

    return a0, a1


def get_longest_two_monotonic_intervals(phase, com):
    intervals = list(get_monotonic_intervals(phase, com))
    lengths = [len(x[0]) for x in intervals]

    *_, isecond, ifirst = np.argpartition(lengths, kth=len(lengths) - 1)

    # # Get first and second longest intervals
    # first, = np.where(np.argsort(lengths) == 0)
    # second, = np.where(np.argsort(lengths) == 1)

    return intervals[ifirst], intervals[isecond]


def get_truncated_longest_sections(phase, com, com_window_size):
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)
    phi1, com1 = longest
    phi2, com2 = second_longest

    com1_mid = com1.mean()
    com2_mid = com2.mean()

    com1_mid = 2330 / 2
    com2_mid = 2330 / 2

    mask1 = (com1 > (com1_mid - com_window_size)) & (
        com1 < (com1_mid + com_window_size)
    )
    mask2 = (com2 > (com2_mid - com_window_size)) & (
        com2 < (com2_mid + com_window_size)
    )

    return ((phi1[mask1], com1[mask1]), (phi2[mask2], com2[mask2]))


def plot_truncated_longest_sections(phase, com, zero_crossing=None, ax=None, com_window_size=None):
    # phasef, comf = get_longest_falling_interval(phase, com)
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)

    # if ax is None:
    #     fig, ax = plt.subplots()

    i0 = np.argmin(np.abs(longest[1] - zero_crossing))
    i1 = np.argmin(np.abs(second_longest[1] - zero_crossing))

    first_zero_crossing = longest[0][i0-5:i0+5], longest[1][i0-5:i0+5]
    second_zero_crossing = second_longest[0][i1-5:i1+5], second_longest[1][i0-5:i0+5]    

    return first_zero_crossing, second_zero_crossing


def get_zero_crossings(phase, com, zero_crossing=None):
    # phasef, comf = get_longest_falling_interval(phase, com)
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)

    # if ax is None:
    #     fig, ax = plt.subplots()

    i0 = np.argmin(np.abs(longest[1] - zero_crossing))
    i1 = np.argmin(np.abs(second_longest[1] - zero_crossing))

    first_zero_crossing = longest[0][i0-5:i0+5], longest[1][i0-5:i0+5]
    second_zero_crossing = second_longest[0][i1-5:i1+5], second_longest[1][i0-5:i0+5]    

    return first_zero_crossing, second_zero_crossing


def get_zero_crossing_slopes(phase, com, zero_crossing=None):
    w = 5
    phases_smoothed, ycoms_smoothed = smooth(phase, com, window=w)

    first_crossing, second_crossing = get_zero_crossings(
        phases_smoothed,
        ycoms_smoothed,
        zero_crossing=1027,
    )

    phase0, pxcom0 = first_crossing
    phase1, pxcom1 = second_crossing

    time0 = phase0 / (360 * 3e9 * 1e-12) # ps
    time1 = phase1 / (360 * 3e9 * 1e-12) # ps
    mcom0 = pxcom0 * 13.7369*1e-6
    mcom1 = pxcom1 * 13.7369*1e-6    
    
    _, m1 = linear_fit(time0, mcom0)
    _, m2 = linear_fit(time1, mcom1)    
    return m1, m2
        

def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = CalibrationMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
