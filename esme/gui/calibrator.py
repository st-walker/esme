from dataclasses import dataclass
import sys
from pathlib import Path
import logging
import time
from typing import Optional
from datetime import datetime

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
from matplotlib import cm
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit

from esme.gui.ui import calibration
from esme.gui.common import (get_tds_calibration_config_dir,
                             setup_screen_display_widget,
                             make_default_i1_lps_machine,
                             make_default_b2_lps_machine)
from esme.calibration import TDSCalibration
from esme.control.configs import load_calibration
from esme.image import filter_image
import logging
import oxfel
from ocelot.cpbd.elements import Quadrupole
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from esme.calibration import calculate_voltage

LOG = logging.getLogger(__name__)

def com_label(ax):
    ax.set_ylabel(r"$y_\mathrm{com}$")


def phase_label(ax):
    ax.set_xlabel(r"$\phi$ / deg")


@dataclass
class PhaseScanSetpoint:
    processed_image: np.ndarray
    phase: float
    amplitude: float
    centre_of_mass: float
    zero_crossing: bool = False

# @dataclass
# class AmplitudeCalibrationSetpoint:
#     r34: float
#     phase_subsample: list[float]
#     com_subsample: list[float]
#     amplitude: float

@dataclass
class AmplitudeCalibrationSetpoint:
    amplitude: float
    voltage: float

@dataclass
class HumanReadableCalibrationData:
    first_zero_crossing: tuple[list, list]
    second_zero_crossing: tuple[list, list]
    phis: list[float]
    ycoms: list[float]
    amplitude: float



@dataclass
class TaggedCalibration:
    calibration: TDSCalibration
    filename: Optional[str] = None
    datetime: Optional[datetime] = None


class CalibrationMainWindow(QMainWindow):
    calibration_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = calibration.Ui_MainWindow()
        self.ui.setupUi(self)

        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

        self.ui.start_calib_button.clicked.connect(self.do_calibration)
        self.ui.load_calib_button.clicked.connect(self.load_calibration_file)
        self.ui.apply_calib_button.clicked.connect(self.apply_calibration_button)

        self.image_plot = setup_screen_display_widget(self.ui.processed_image_plot)

        self.com_scatter = make_pixel_widths_scatter(self.ui.centre_of_mass_with_phase_plot,
                                                     title="Centre of Mass vs Phase",
                                                     xlabel="Phi",
                                                     xunits="Degrees",
                                                     ylabel="Centre of Mass",
                                                     yunits="m")


        self.calibration = None
        self.voltages = []
        self.amplitudes = []

        self.setup_plots()

    def do_calibration(self):
        self.worker = CalibrationWorker(self.machine,
                                        screen_name=self.ui.screen_name_line_edit.text(),
                                        amplitudes=self.parse_amplitude_input_box(),
                                        cut=self.ui.cut_index_spinbox.value())
        self.thread = QThread()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.processed_image_signal.connect(self.post_phase_scan_increment)
        self.worker.hrcd_signal.connect(self.post_human_readable_calibration_data)
        self.worker.amplitude_setpoint_signal.connect(self.post_calib_setpoint_result)
        self.thread.start()


    def post_phase_scan_increment(self, phase_scan_increment: PhaseScanSetpoint):
        image = phase_scan_increment.processed_image
        phase = phase_scan_increment.phase
        amplitude = phase_scan_increment.amplitude
        centre_of_mass = phase_scan_increment.centre_of_mass
        is_zero_crossing = phase_scan_increment.zero_crossing

        # from IPython import embed; embed()phase

        self.com_scatter.addPoints([phase], [centre_of_mass])

        if is_zero_crossing:
            return

        items = self.image_plot.items
        image_item = items[0]
        image_item.setImage(image)

    @property
    def tds(self):
        return self.machine.deflectors[DiagnosticRegion.I1]
        return self.machine.deflectors.active_tds()

    def setup_plots(self):
        ax1 = self.ui.zero_crossing_extraction_plot.axes

        # ax1 = self.ui.phase_com_plot_widget.axes
        # ax2 = self.ui.amp_voltage_plot_widget.axes
        com_label(ax1)
        phase_label(ax1)
        ax1.set_title("Center of Mass against TDS Phase")

        # ax2.set_xlabel("Amplitude / %")
        # ax2.set_ylabel("Voltage / MV")
        # ax2.set_title("TDS Calibration")

    def post_human_readable_calibration_data(self, hrcd):
        first_zero_crossing = hrcd.first_zero_crossing
        second_zero_crossing = hrcd.second_zero_crossing
        phis = hrcd.phis
        ycoms = hrcd.ycoms
        amplitude = hrcd.amplitude

        # from IPython import embed; embed()
        ax = self.ui.zero_crossing_extraction_plot.axes
        # com_label(ax)
        # phase_label(ax)

        ax.plot(phis, ycoms, label=f"Amplitude = {amplitude}%", alpha=0.5)
        ax.plot(*first_zero_crossing, color="black")
        ax.plot(*second_zero_crossing, color="black")
        ax.legend()
        self.ui.zero_crossing_extraction_plot.draw()
        # ax3.set_ylabel("Streak $\mathrm{\mu{}m\,/\,ps}$")

    def post_calib_setpoint_result(self, calib_sp):
        amplitude = calib_sp.amplitude
        voltage = calib_sp.voltage
        ax = self.ui.final_calibration_plot.axes
        ax.scatter(amplitude, voltage)

        self.voltages.append(voltage)
        self.amplitudes.append(amplitude)

        self.ui.final_calibration_plot.draw()
        ax.set_xlabel("Amplitude / %")
        ax.set_ylabel("Voltage / V")

    def parse_amplitude_input_box(self):
        text = self.ui.amplitudes_line_edit.text()
        return np.array([float(y) for y in text.split(",")])

    def update_calibration_plots(self):
        voltages = self.calibration.get_voltages()
        amplitudes = self.calibration.get_amplitudes()
        ax = self.ui.amp_voltage_plot_widget.axes
        ax.clear()
        # self.setup_plots()
        ax.scatter(amplitudes, voltages * 1e-6)

        # mapping = self.calibration.get_calibration_mapping()
        # sample_ampls, fit_voltages = mapping.get_voltage_fit_line()
        # m, c = mapping.get_voltage_fit_parameters()

        # label = fr"Fit: $m={abs(m)*1e-6:.3f}\mathrm{{MV}}\,/\,\%$, $c={c*1e-6:.2f}\,\mathrm{{MV}}$"
        # ax.plot(sample_ampls, fit_voltages*1e-6)
        # ax.legend()
        self.draw()

    # def get_plot_widgets(self):
    #     return [self.ui.amp_voltage_plot_widget,
    #             self.ui.phase_com_plot_widget,
    #             self.ui.phase_com_plot_widget]

    # def draw(self):
    #     for wi in self.get_plot_widgets():
    #         wi.draw()

    def load_calibration_file(self):
        options = QFileDialog.Options()
        outdir = get_tds_calibration_config_dir() / "i1"
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
    processed_image_signal = pyqtSignal(PhaseScanSetpoint)
    amplitude_setpoint_signal = pyqtSignal(AmplitudeCalibrationSetpoint)
    hrcd_signal = pyqtSignal(HumanReadableCalibrationData)

    def __init__(self, machine, screen_name, amplitudes, cut):
        super().__init__()
        self.machine = machine
        self.screen_name = screen_name
        # XXX Why is this hardcoded?!
        self.screen_name = "OTRC.64.I1D"
        self.amplitudes = amplitudes
        self.cut = cut
        self.cut = 10000000
        self.kill = False

    def get_image(self):
        image = self.machine.screens.get_image(self.screen_name)
        LOG.info("Reading image from: %s", self.screen_name)
        return image # .astype(np.float32)

    def calibrate(self):
        self.machine.deflectors.active_tds().set_amplitude(0.0)
        time.sleep(1.0)
        image = self.machine.screens.get_image_raw(self.screen_name)
        image = image[:self.cut]
        com = ndi.center_of_mass(image)
        ycom = com[1]
        yzero_crossing = ycom

        slopes = []
        for amplitude in self.amplitudes:
            self.machine.deflectors.active_tds().set_amplitude(amplitude)
            m1, m2 = self.calibrate_once(yzero_crossing, amplitude)
            slopes.append((np.mean(abs(m1[0])), np.mean(abs(m2[0]))))

    def get_r34_to_screen(self):
        lat = oxfel.cat_to_i1d()
        screen_name = self.screen_name
        subseq = lat.get_sequence(start="TDSA.52.I1",
                                  stop=screen_name)
        subseq[0].l /= 2
        quads = [ele for ele in subseq if isinstance(ele, Quadrupole)]
        for quad in quads:
            k1l_mrad = self.machine.scanner.get_quad_strength(quad.id)
            k1l_rad = k1l_mrad * 1e-3
            quad.k1l = k1l_rad

        lat = MagneticLattice(subseq)
        energy_mev = 130
        _, rmat, _ = lat.transfer_maps(energy_mev)
        r34 = rmat[2, 3]

        print(f"R34 = {r34}")
        return r34

    def calibrate_once(self, zero_crossing, amplitude):
        phis = np.linspace(-180, 200, num=191)
        # phis = np.linspace(-180, 200, num=15)
        ycoms = []
        tds = self.machine.deflectors.active_tds()
        tds.set_phase(phis[0])
        time.sleep(4)
        for phi in phis:
            time.sleep(0.25)
            tds.set_phase(phi)
            image = self.machine.screens.get_image_raw(self.screen_name)
            image = image[:self.cut]
            image = filter_image(image, 0)
            com = ndi.center_of_mass(image)
            ycom = com[1]
            ycoms.append(ycom)
            sp = PhaseScanSetpoint(processed_image=image,
                                   phase=phi,
                                   amplitude=amplitude,
                                   centre_of_mass=ycom,
                                   zero_crossing=False)

            self.processed_image_signal.emit(sp)

        first_zero_crossing, second_zero_crossing = get_zero_crossings(phis, ycoms, zero_crossing=zero_crossing)

        m1, m2 = get_zero_crossing_slopes(phis, ycoms, zero_crossing=zero_crossing)

        hrcd = HumanReadableCalibrationData(first_zero_crossing, second_zero_crossing,
                                            phis,
                                            ycoms,
                                            amplitude=amplitude)

        self.hrcd_signal.emit(hrcd)

        m = np.mean([abs(m1[0]), abs(m2[0])])

        m_m_per_s = m * (360 * 3e9) / 13.7369*1e-6
        phase0, pxcom0 = first_zero_crossing
        phase1, pxcom1 = second_zero_crossing

        time0 = phase0 / (360 * 3e9 * 1e-12) # ps
        time1 = phase1 / (360 * 3e9 * 1e-12) # ps
        mcom0 = pxcom0 * 13.7369*1e-6
        mcom1 = pxcom1 * 13.7369*1e-6

        r34 = self.get_r34_to_screen()
        m = abs(np.gradient(mcom0, time0)[5] * 1e12)  # from per ps to s


        voltage_v = calculate_voltage(slope=m, r34=r34, energy=130, frequency=3e9)

        print(amplitude, voltage_v)
        calib_sp = AmplitudeCalibrationSetpoint(amplitude, voltage_v)
        self.amplitude_setpoint_signal.emit(calib_sp)


        return m1, m1 #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX JUST USING M1 HERE!!!...

        from IPython import embed; embed()
        # time_s = phi / (3e9 * 360)
        # ycoms_m = np.array(ycoms) * 13.7369 * 1e-6

        m1, m2 = get_zero_crossing_slopes(phis, ycoms, zero_crossing=zero_crossing)
        return m1, m2 #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX JUST USING M1 HERE!!!...

    def run(self):
        slopes = self.calibrate()

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
    a0 = popt[0], perr[0] # y-intercept with error
    a1 = popt[1], perr[1] # gradient with error

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


def get_zero_crossings(phase, com, zero_crossing=None):
    # phasef, comf = get_longest_falling_interval(phase, com)
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)

    # if ax is None:
    #     fig, ax = plt.subplots()

    i0 = np.argmin(np.abs(longest[1] - zero_crossing))
    i1 = np.argmin(np.abs(second_longest[1] - zero_crossing))

    first_zero_crossing = longest[0][i0-5:i0+5], longest[1][i0-5:i0+5]
    second_zero_crossing = second_longest[0][i1-5:i1+5], second_longest[1][i1-5:i1+5]

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




def make_pixel_widths_scatter(widget, title, xlabel, xunits, ylabel, yunits):
    plot = widget.addPlot(title=title)

    plot.setLabel('bottom', xlabel, units=xunits)
    plot.setLabel('left', ylabel, units=yunits)

    scatter = pg.ScatterPlotItem()
    plot.addItem(scatter)

    return scatter


def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = CalibrationMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
