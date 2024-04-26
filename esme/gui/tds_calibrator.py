from dataclasses import dataclass
import sys
import logging
import time
from typing import Optional, Union, Iterator
from datetime import datetime
import os
from pathlib import Path
from typing import TypeVar, Callable
from types import SimpleNamespace

import csv
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow, QTableWidgetItem
from PyQt5 import QtWidgets
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
import oxfel
from ocelot.cpbd.elements import Quadrupole
from ocelot.cpbd.magnetic_lattice import MagneticLattice

from esme.load import load_calibration_from_yaml
from esme.gui.ui import calibration
from esme.gui.widgets.common import (get_tds_calibration_config_dir,
                             setup_screen_display_widget,
                             make_default_i1_lps_machine,
                             make_default_b2_lps_machine,
                             make_default_injector_espread_machine)
from esme.control.configs import load_calibration
from esme.gui.widgets.area import AreaControl
from esme.calibration import CompleteCalibration
from esme.image import filter_image
from esme.calibration import calculate_voltage, AmplitudeVoltageMapping
from esme import DiagnosticRegion

LayoutBaseType = TypeVar("LayoutBaseType", bound=QtWidgets.QLayout)


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

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
    calibration: AmplitudeVoltageMapping
    filename: Optional[str] = None
    datetime: Optional[datetime] = None


class CalibrationMainWindow(QMainWindow):
    avmapping_signal = pyqtSignal(AmplitudeVoltageMapping)
    NROWS = 100
    NCOLS = 7
    # When measuring the crossing, we take data at the lower and upper
    # phases, but also in between.  The total amount of phase setpoints is set here.
    # E.g. 5 = the two ends plus 3 the middle.
    PHASE_COMS_TO_MEASURE = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui(parent=parent)

        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

    def init_ui(self, fname=None, parent=None):
        self.setWindowTitle('Bolko Redux Mk. II')
        self.updating_table = False  # Flag to prevent recursive updates

        self.ui = SimpleNamespace()

        self.ui.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.ui.main_widget)
        main_layout = QtWidgets.QHBoxLayout(self.ui.main_widget)

        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)
        self.create_area_group_box(left_layout)

        self.create_calibration_group_box(left_layout)

        self.ui.file_path_label = QtWidgets.QLabel()
        self.ui.load_button = QtWidgets.QPushButton("Load...")
        self.ui.load_button.clicked.connect(self.load_yaml_file)

        io_layout = QtWidgets.QHBoxLayout()
        left_layout.addLayout(io_layout)

        load_layout = QtWidgets.QHBoxLayout()
        load_layout.addWidget(self.ui.load_button)
        load_layout.addWidget(self.ui.file_path_label)
        io_layout.addLayout(load_layout)

        run_apply_layout = QtWidgets.QHBoxLayout()
        left_layout.addLayout(run_apply_layout)

        self.ui.apply_button = QtWidgets.QPushButton("Apply")
        self.ui.apply_button.clicked.connect(self.apply_calibration)
        # Add the button underneath the table

        run_apply_layout.addWidget(self.ui.apply_button)
        self.ui.start_calib_button = QtWidgets.QPushButton("Start Calibration")
        self.ui.start_calib_button.clicked.connect(self.do_calibration)
        run_apply_layout.addWidget(self.ui.start_calib_button)


        # if parent is None:
            # self.ui.apply_button.hide()

        self.ui.graphics_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.ui.graphics_widget)

        self.create_plots(self.ui.graphics_widget)
        self.create_table(left_layout)

        save_layout = QtWidgets.QHBoxLayout()
        self.ui.save_button = QtWidgets.QPushButton("Save...")
        self.ui.save_button.clicked.connect(self.save_table_data)
        save_layout.addWidget(self.ui.save_button)
        io_layout.addLayout(save_layout)

        self.set_filepath_label(fname)

    def do_calibration(self) -> None:
        for amplitude, phase_pair0, phase_pair1 in self._iter_rows():
            self.machine.deflector.set_amplitude(amplitude)
            self._scan_one_phase_pair(*phase_pair0)
            self._scan_one_phase_pair(*phase_pair1)


    def _scan_one_phase_pair(self, low, high) -> None:
        phases = np.linspace(low, high, num=self.PHASE_COMS_TO_MEASURE)
        for phase in phases:
            self.machine.deflector.set_phase(phase)
            # image = self.machine.screen.get_image_raw() ???
            com = self.get_com_from_screen()

    def _get_com_from_screen(self) -> tuple[float, float]:
        pass

    def _iter_rows(self) -> Iterator[tuple[float, tuple[float, float], tuple[float, float]]]:
        for irow in range(self.ui.table_widget.rowCount()):
            amplitude = self.ui.table_widget.item(irow, 0)
            phase_pair0 = self.ui.table_widget.item(irow, 1), self.ui.table_widget.item(irow, 2)
            phase_pair1 = self.ui.table_widget.item(irow, 3), self.ui.table_widget.item(irow, 4)
            yield amplitude, phase_pair0, phase_pair1

    def get_amplitude_voltage_mapping(self) -> AmplitudeVoltageMapping:
        amplitudes = []
        voltages = []
        for irow in range(self.ui.table_widget.rowCount()):
            try:
                amplitude = self._get_amplitude_row(irow)
                voltage = self._get_voltage_row(irow)
            except (AttributeError, ValueError):
                continue
            else:
                amplitudes.append(amplitude)
                voltages.append(voltage)

        return AmplitudeVoltageMapping(DiagnosticRegion.I1, amplitudes, voltages)

    def _init_gui_from_complete_calib(self, calib, fname: str) -> None:
        self.set_filepath_label(fname)
        self.ui.r_3412_spin_box.setValue(calib.r34_from_optics())
        calfactors = calib.cal_factors #* calib.CAL_UM_PER_PS
        self._write_table_contents(calib.amplitudes, calfactors)

    def _write_table_contents(self, amplitudes: list[float], cal_factors: list[float]) -> None:
        for i, (amp, cal) in enumerate(zip(amplitudes, cal_factors)):
            self.ui.table_widget.setItem(i, 0, QTableWidgetItem(str(amp)))
            self._set_cal_factor_row(i, cal)
        for j in range(i + 1, self.NROWS):
            self.ui.table_widget.setItem(j, 0, QTableWidgetItem(""))
            self.ui.table_widget.setItem(j, 1, QTableWidgetItem(""))
            self.ui.table_widget.setItem(j, 2, QTableWidgetItem(""))
            self.ui.table_widget.setItem(j, 3, QTableWidgetItem(""))
            self.ui.table_widget.setItem(j, 4, QTableWidgetItem(""))

    def create_table(self, left_layout: LayoutBaseType) -> None:
        self.ui.table_widget = QtWidgets.QTableWidget(self.NROWS, self.NCOLS)
        self.ui.table_widget.setHorizontalHeaderLabels(
            ["Amplitude / %",
             "  𝜙₀₀ / °  ",
             "  𝜙₀₁ / °  ",
             "  𝜙₁₀ / °  ",
             "  𝜙₁₁ / °  ",
             "Calibration Factor µm/ps",
             "Voltage / MV"]
        )
        self.ui.table_widget.horizontalHeader().setStretchLastSection(True)
        # self.ui.table_widget.itemChanged.connect(self.update_plots_and_voltage)
        left_layout.addWidget(self.ui.table_widget)

        for row in range(self.ui.table_widget.rowCount()):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.ui.table_widget.setItem(row, 6, item)
        self.ui.table_widget.resizeColumnsToContents()

    def create_area_group_box(self, layout: LayoutBaseType) -> None:
        group_box = QtWidgets.QGroupBox("Diagnostic Area")
        group_layout = QtWidgets.QGridLayout(group_box)
        self.ui.area_control = AreaControl()
        group_layout.addWidget(self.ui.area_control)
        layout.addWidget(group_box)

    def create_calibration_group_box(self, layout: LayoutBaseType) -> None:
        group_box = QtWidgets.QGroupBox("Calibration Machine Setpoint")
        # group_layout = QtWidgets.QVBoxLayout(group_box)
        group_layout = QtWidgets.QGridLayout(group_box)

        self.ui.r_3412_spin_box = self.create_double_spin_box(
            "<span>R<sub>12(34)</sub> / mrad<sup>-1</sup></span>", group_layout,
            default_value=0.0, min_value=-20.0, max_value=20.0, step=1.0,
            read_function=self.read_r1234_from_machine)
        self.ui.beam_energy_spin_box = self.create_double_spin_box(
            "Beam Energy / MeV", group_layout, default_value=130,
            min_value=0.0, max_value=sys.float_info.max, step=1.0,
            read_function=self.read_beam_energy_from_machine)
        self.ui.tds_frequency_spin_box = self.create_double_spin_box(
            "TDS Frequency / GHz", group_layout, default_value=3.0,
            min_value=0.0, max_value=sys.float_info.max, step=1.0)

        layout.addWidget(group_box)

    def read_r1234_from_machine(self) -> None:
        screen_name = self.ui.area_control.get_selected_screen_name()
        r1234 = self.machine.optics.r12_streaking_from_tds_to_point(screen_name)
        self.ui.r_3412_spin_box.setValue(r1234)

    def read_beam_energy_from_machine(self) -> None:
        energy_mev = self.machine.optics.get_beam_energy() # MeV is fine here.
        self.ui.beam_energy_spin_box.setValue(energy_mev)

    def load_yaml_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load YAML File", "", "YAML Files (*.yaml)")
        # if file_path:
        #     self.set_filepath_label(file_path)
        # else:
        #     self.set_filepath_label("")
        if not file_path:
            return
        calibration = load_calibration_from_yaml(file_path)
        self._init_gui_from_complete_calib(calibration, file_path)

    def set_filepath_label(self, file_path: Union[os.PathLike, str]) -> None:
        if not file_path:
            file_path = "No file selected"
        # try to make file path relative to cwd else just use filepath
        try:
            file_path = Path(file_path).relative_to(os.getcwd())
        except ValueError:
            pass
        self.ui.file_path_label.setText(f"File: {file_path}")

    def create_double_spin_box(self, label: str, layout: LayoutBaseType, default_value: float = 0.0,
                               min_value: float = 0.0, max_value: float = 100.0, step: float = 0.1,
                               read_function: Callable[None, float] | None = None) -> None:
        current_row_count = layout.rowCount()

        label_widget = QtWidgets.QLabel(label)
        label_widget.setTextFormat(QtCore.Qt.RichText)

        layout.addWidget(label_widget, current_row_count, 0)

        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setMinimum(min_value)
        spin_box.setMaximum(max_value)
        spin_box.setSingleStep(step)
        spin_box.setValue(default_value)
        # layout.addWidget(spin_box)
        layout.addWidget(spin_box, current_row_count, 1)
        # layout.addWidget(container)
        if read_function:
            new_button = QtWidgets.QPushButton("Read From Machine")
            new_button.clicked.connect(read_function)
            layout.addWidget(new_button, current_row_count, 2)
        return spin_box

    def create_phase_scan_plot(self, graphics_widget: pg.GraphicsLayoutWidget) -> None:
        phase_scan_plot = graphics_widget.addPlot(title="Phase Scans", row=1, col=0)
        phase_scan_plot.setLabel('bottom', 'TDS Phase / °')
        phase_scan_plot.setLabel('right', 'Centre of Mass / px')
        self.ui.phase_scan_plot = phase_scan_plot
        phase_scan_plot_data_item = pg.PlotDataItem()

    def create_calibration_plot(self, graphics_widget: pg.GraphicsLayoutWidget) -> None:
        self.ui.cal_factors_plot = graphics_widget.addPlot(title="TDS Calibration Factors")
        self.ui.cal_factors_plot.setLabel('bottom', 'TDS Amplitude')
        self.cal_factor_plot_data_item = pg.PlotDataItem()
        self.ui.cal_factors_plot.addItem(self.cal_factor_plot_data_item)

        self.ui.cal_factors_plot.getAxis('left').setLabel('Calibration Factor µm/ps', color='#00FFFF')
        # self.ui.cal_factors_plot.setLabels(left='Calibration Factor µm/ps')

        ## create a new ViewBox, link the right axis to its coordinate system
        self.ui.tds_voltage_plot = pg.ViewBox()
        self.ui.cal_factors_plot.showAxis('right')
        self.ui.cal_factors_plot.scene().addItem(self.ui.tds_voltage_plot)
        self.ui.cal_factors_plot.getAxis('right').linkToView(self.ui.tds_voltage_plot)
        self.ui.tds_voltage_plot.setXLink(self.ui.cal_factors_plot)
        self.ui.cal_factors_plot.getAxis('right').setLabel('TDS Voltage / MV', color='#FFFF00')

        self.ui.tds_voltage_plot.setGeometry(self.ui.cal_factors_plot.vb.sceneBoundingRect())
        self.ui.tds_voltage_plot.linkedViewChanged(self.ui.cal_factors_plot.vb, self.ui.tds_voltage_plot.XAxis)

        self.ui.cal_factors_plot.plot([1, 2, 4, 8, 16, 32], pen="#00FFFF")
        self.ui.tds_voltage_plot.addItem(pg.PlotCurveItem([10, 20, 40, 80, 40, 20], pen='#FFFF00'))

        self.ui.cal_factors_plot.vb.sigResized.connect(self._update_views)

    def create_plots(self, graphics_widget: pg.GraphicsLayoutWidget) -> None:
        self.create_calibration_plot(graphics_widget)
        self.create_phase_scan_plot(graphics_widget)


    def _update_views(self) -> None:
        self.ui.tds_voltage_plot.setGeometry(self.ui.cal_factors_plot.vb.sceneBoundingRect())
        self.ui.tds_voltage_plot.linkedViewChanged(self.ui.cal_factors_plot.vb, self.ui.tds_voltage_plot.XAxis)

    def update_plots_and_voltage(self, _) -> None:
        if self.updating_table:
            return
        self.updating_table = True

        for row in range(self.ui.table_widget.rowCount()):
            try:
                calibration_item = self.ui.table_widget.item(row, 1)
                cal_factor = float(calibration_item.text()) / CompleteCalibration.CAL_UM_PER_PS
                r = self.ui.r_3412_spin_box.value()
                energy = self.ui.beam_energy_spin_box.value()
                frequency_ghz = self.ui.tds_frequency_spin_box.value()
                frequency_hz = frequency_ghz * 1e9
                voltage_v = calculate_voltage(slope=cal_factor, r34=r, energy=energy, frequency=frequency_hz)
            except (TypeError, ZeroDivisionError, AttributeError, ValueError):
                self._set_voltage_row(row, None)
            else:
                self._set_voltage_row(row, voltage_v)
                # voltage_mv = voltage_v * 1e-6
                # self.ui.table_widget.item(row, 2).setText(f"{voltage_mv:.4g}")

        self.update_plots()
        self.updating_table = False

    def apply_calibration(self) -> None:
        self.avmapping_signal.emit(self.get_amplitude_voltage_mapping())

    def _set_cal_factor_row(self, irow: int, cal_factor_m_per_s: float) -> None:
        calfactor = cal_factor_m_per_s * CompleteCalibration.CAL_UM_PER_PS
        self.ui.table_widget.setItem(irow, 1, QTableWidgetItem(f"{calfactor:.4g}"))

    def _set_voltage_row(self, irow: int, voltage_v: float) -> None:
        if voltage_v is None:
            try:
                self.ui.table_widget.item(irow, 4).setText("")
            except AttributeError:
                pass
        else:
            voltage_mv = voltage_v * 1e-6
            self.ui.table_widget.item(irow, 2).setText(f"{voltage_mv:.4g}")

    def _get_voltage_row(self, irow: int) -> float:
        voltage_item = self.ui.table_widget.item(irow, 2)
        return float(voltage_item.text()) * 1e6

    def _get_amplitude_row(self, irow: int) -> float:
        amplitude_item = self.ui.table_widget.item(irow, 0)
        return float(amplitude_item.text())

    def update_plots(self) -> None:
        amplitudes = []
        calibrations = []
        voltages = []
        for row in range(self.ui.table_widget.rowCount()):
            try:
                amplitude = float(self.ui.table_widget.item(row, 0).text())
                calibration = float(self.ui.table_widget.item(row, 1).text())
                # amplitude.append(float(self.ui.table_widget.item(row, 0).text()))
                # calibration.append(float(self.ui.table_widget.item(row, 1).text()))
                voltage = float(self.ui.table_widget.item(row, 2).text())
                # voltage.append(float(self.ui.table_widget.item(row, 2).text()))
            except (TypeError, ValueError, AttributeError):
                continue
            else:
                amplitudes.append(amplitude)
                calibrations.append(calibration)
                voltages.append(voltage)

        self.cal_factor_plot_data_item.setData(amplitudes, calibrations)
        self.tds_voltage_plot_data_item.setData(amplitudes, voltages)

    def save_table_data(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Table Data", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write headers
                headers = [self.ui.table_widget.horizontalHeaderItem(i).text() for i in range(self.ui.table_widget.columnCount())]
                writer.writerow(headers)
                # Write data rows
                for row in range(self.ui.table_widget.rowCount()):
                    row_data = []
                    for column in range(self.ui.table_widget.columnCount()):
                        item = self.ui.table_widget.item(row, column)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)


class CalibrationMainWindowOld(QMainWindow):
    avmapping_signal = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = calibration.Ui_MainWindow()
        self.ui.setupUi(self)

        self.i1machine = make_default_injector_espread_machine() # make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine

        self.ui.start_calib_button.clicked.connect(self.do_calibration)
        self.ui.load_calib_button.clicked.connect(self.load_calibration_file)
        self.ui.apply_calib_button.clicked.connect(self.ui.apply_calibration_button)

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
                                        amplitudes=self.parse_amplitude_input_box())
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
        phase_scan_increment.amplitude
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
        # return self.machine.deflectors[DiagnosticRegion.I1]
        return self.machine.deflector
        # return self.machine.deflectors.active_tds()

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
        self.ui.apply_calibration_button.click()

    def apply_calibration_button(self):
        if self.calibration is not None:
            mapping = self.get_amplitude_voltage_mapping()
            self.calibration_signal.emit(mapping)


class CalibrationWorker(QObject):
    processed_image_signal = pyqtSignal(PhaseScanSetpoint)
    amplitude_setpoint_signal = pyqtSignal(AmplitudeCalibrationSetpoint)
    hrcd_signal = pyqtSignal(HumanReadableCalibrationData)

    def __init__(self, machine, screen_name, amplitudes):
        super().__init__()
        self.machine = machine
        self.screen_name = screen_name
        # XXX Why is this hardcoded?!
        self.screen_name = "OTRC.64.I1D"
        self.amplitudes = amplitudes
        self.kill = False

    def get_image(self):
        image = self.machine.screen.get_image_raw()
        LOG.info("Reading image from: %s", self.screen_name)
        return image # .astype(np.float32)

    def calibrate(self):
        # self.machine.deflectors.active_tds().set_amplitude(0.0)
        self.machine.deflector.set_amplitude(0.0)
        time.sleep(1.0)
        print(self.machine.screen, self.machine.screen.name, "!!!!!!!!!!!!")
        image = self.machine.screen.get_image_raw()
        com = ndi.center_of_mass(image)
        ycom = com[1]
        yzero_crossing = ycom

        slopes = []
        for amplitude in self.amplitudes:
            # self.machine.deflectors.active_tds().set_amplitude(amplitude)
            self.machine.deflector.set_amplitude(amplitude)
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

        return r34

    def calibrate_once(self, zero_crossing, amplitude):
        phis = np.linspace(-180, 200, num=191)
        # phis = np.linspace(-180, 200, num=15)
        ycoms = []
        tds = self.machine.deflector
        tds.set_phase(phis[0])
        time.sleep(4)
        for phi in phis:
            time.sleep(0.25)
            tds.set_phase(phi)
            image = self.machine.screen.get_image_raw()
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

        m * (360 * 3e9) / 13.7369*1e-6
        phase0, pxcom0 = first_zero_crossing
        phase1, pxcom1 = second_zero_crossing

        time0 = phase0 / (360 * 3e9 * 1e-12) # ps
        time1 = phase1 / (360 * 3e9 * 1e-12) # ps
        mcom0 = pxcom0 * 13.7369*1e-6
        pxcom1 * 13.7369*1e-6

        r34 = self.get_r34_to_screen()
        m = abs(np.gradient(mcom0, time0)[5] * 1e12)  # from per ps to s

        voltage_v = calculate_voltage(slope=m, r34=r34, energy=130, frequency=3e9)

        print(amplitude, voltage_v)
        calib_sp = AmplitudeCalibrationSetpoint(amplitude, voltage_v)
        self.amplitude_setpoint_signal.emit(calib_sp)


        return m1, m1 #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX JUST USING M1 HERE!!!...

        # from IPython import embed; embed()
        # time_s = phi / (3e9 * 360)
        # ycoms_m = np.array(ycoms) * 13.7369 * 1e-6

        m1, m2 = get_zero_crossing_slopes(phis, ycoms, zero_crossing=zero_crossing)
        return m1, m2 # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX JUST USING M1 HERE!!!...

    def run(self):
        self.calibrate()

    def update_screen_name(self, screen_name):
        LOG.info(f"Setting screen name for Screen Worker thread: {screen_name}")
        self.screen_name = screen_name


def smooth(phase, com, window):
    w = window
    ycoms_smoothed = np.convolve(com, np.ones(w), "valid") / w
    phases_smoothed = np.convolve(phase, np.ones(w), "valid") / w
    return phases_smoothed, ycoms_smoothed


def get_monotonic_intervals(phases, coms):
    # We want to find where the centres of mass are consistently
    # rising and falling.
    deriv = np.diff(coms)

    rising_mask = deriv > 0
    falling_mask = deriv <= 0

    (indices,) = np.where(np.diff(rising_mask))
    indices += 1  # off by one otherwise.

    phases[indices]

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



class CalibrationExplorer(QtWidgets.QMainWindow):
    avmapping_signal = pyqtSignal(AmplitudeVoltageMapping)
    NROWS = 100

    def __init__(self, ccalib: CompleteCalibration, fname=None, parent=None):
        super().__init__(parent)
        self.init_ui(ccalib, fname=fname, parent=parent)

    def init_ui(self, ccalib, fname=None, parent=None):
        self.setWindowTitle('TDS Calibration Explorer')
        self.updating_table = False  # Flag to prevent recursive updates

        self.ui.main_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.ui.main_widget)
        main_layout = QtWidgets.QHBoxLayout(self.ui.main_widget)

        left_layout = QtWidgets.QVBoxLayout()
        main_layout.addLayout(left_layout)

        self.create_calibration_group_box(left_layout)

        self.ui.file_path_label = QtWidgets.QLabel()
        self.ui.load_button = QtWidgets.QPushButton("Load...")
        self.ui.load_button.clicked.connect(self.load_yaml_file)

        io_layout = QtWidgets.QHBoxLayout()
        left_layout.addLayout(io_layout)

        load_layout = QtWidgets.QHBoxLayout()
        load_layout.addWidget(self.ui.load_button)
        load_layout.addWidget(self.ui.file_path_label)
        io_layout.addLayout(load_layout)

        self.ui.apply_button = QtWidgets.QPushButton("Apply")
        self.ui.apply_button.clicked.connect(self.apply_calibration)
        # Add the button underneath the table
        left_layout.addWidget(self.ui.apply_button)
        if parent is None:
            self.ui.apply_button.hide()

        self.ui.graphics_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self.ui.graphics_widget)

        self.create_plots()
        self.create_table(left_layout)

        save_layout = QtWidgets.QHBoxLayout()
        self.ui.save_button = QtWidgets.QPushButton("Save...")
        self.ui.save_button.clicked.connect(self.save_table_data)  # Connect to the slot for saving
        save_layout.addWidget(self.ui.save_button)  # Assuming load_layout is your QHBoxLayout
        io_layout.addLayout(save_layout)

        self.set_filepath_label(fname)

        if ccalib:
            self._init_gui_from_complete_calib(ccalib, fname)

    def get_amplitude_voltage_mapping(self):
        amplitudes = []
        voltages = []
        for irow in range(self.ui.table_widget.rowCount()):
            try:
                amplitude = self._get_amplitude_row(irow)
                voltage = self._get_voltage_row(irow)
            except (AttributeError, ValueError):
                continue
            else:
                amplitudes.append(amplitude)
                voltages.append(voltage)

        return AmplitudeVoltageMapping(DiagnosticRegion.I1, amplitudes, voltages)

    def _init_gui_from_complete_calib(self, calib, fname):
        self.set_filepath_label(fname)
        self.ui.r_3412_spin_box.setValue(calib.r34_from_optics())
        calfactors = calib.cal_factors #* calib.CAL_UM_PER_PS
        self._write_table_contents(calib.amplitudes, calfactors)

    def _write_table_contents(self, amplitudes, cal_factors):
        for i, (amp, cal) in enumerate(zip(amplitudes, cal_factors)):
            self.ui.table_widget.setItem(i, 0, QTableWidgetItem(str(amp)))
            self._set_cal_factor_row(i, cal)
        for j in range(i + 1, self.NROWS):
            self.ui.table_widget.setItem(j, 0, QTableWidgetItem(""))
            self.ui.table_widget.setItem(j, 1, QTableWidgetItem(""))

    def create_table(self, left_layout):
        self.ui.table_widget = QtWidgets.QTableWidget(self.NROWS, 3)
        self.ui.table_widget.setHorizontalHeaderLabels(
            ["Amplitude / %", "Calibration Factor µm/ps", "Voltage / MV"])
        self.ui.table_widget.horizontalHeader().setStretchLastSection(True)
        self.ui.table_widget.itemChanged.connect(self.update_plots_and_voltage)
        left_layout.addWidget(self.ui.table_widget)

        for row in range(self.ui.table_widget.rowCount()):
            item = QtWidgets.QTableWidgetItem()
            item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
            self.ui.table_widget.setItem(row, 2, item)
        self.ui.table_widget.resizeColumnsToContents()

    def create_calibration_group_box(self, layout):
        group_box = QtWidgets.QGroupBox("Calibration Machine Setpoint")
        group_layout = QtWidgets.QVBoxLayout(group_box)

        self.ui.r_3412_spin_box = self.create_double_spin_box(
            "<span>R<sub>12(34)</sub> / mrad<sup>-1</sup></span>", group_layout,
            default_value=0.0, min_value=-20.0, max_value=20.0, step=1.0)
        self.ui.tds_frequency_spin_box = self.create_double_spin_box(
            "TDS Frequency / GHz", group_layout, default_value=3.0,
            min_value=0.0, max_value=sys.float_info.max, step=1.0)
        self.ui.beam_energy_spin_box = self.create_double_spin_box(
            "Beam Energy / MeV", group_layout, default_value=130,
            min_value=0.0, max_value=sys.float_info.max, step=1.0)

        layout.addWidget(group_box)

    def load_yaml_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load YAML File", "", "YAML Files (*.yaml)")
        # if file_path:
        #     self.set_filepath_label(file_path)
        # else:
        #     self.set_filepath_label("")
        if not file_path:
            return
        calibration = load_calibration_from_yaml(file_path)
        self._init_gui_from_complete_calib(calibration, file_path)

    def set_filepath_label(self, file_path: Union[os.PathLike, str]):
        if not file_path:
            file_path = "No file selected"
        # try to make file path relative to cwd else just use filepath
        try:
            file_path = Path(file_path).relative_to(os.getcwd())
        except ValueError:
            pass
        self.ui.file_path_label.setText(f"File: {file_path}")

    def create_double_spin_box(self, label, layout, default_value=0.0,
                               min_value=0.0, max_value=100.0, step=0.1):
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QHBoxLayout(container)
        label_widget = QtWidgets.QLabel(label)
        label_widget.setTextFormat(QtCore.Qt.RichText)
        container_layout.addWidget(label_widget)
        spin_box = QtWidgets.QDoubleSpinBox()
        spin_box.setMinimum(min_value)
        spin_box.setMaximum(max_value)
        spin_box.setSingleStep(step)
        spin_box.setValue(default_value)
        container_layout.addWidget(spin_box)
        layout.addWidget(container)
        return spin_box

    def create_plots(self):
        self.ui.cal_factors_plot = self.ui.graphics_widget.addPlot(title="TDS Calibration Factors")
        # self.ui.cal_factors_plot.setLabel('bottom', 'TDS Amplitude')
        self.ui.cal_factors_plot.setLabel('left', 'Calibration Factor µm/ps')
        self.cal_factor_plot_data_item = pg.PlotDataItem()
        self.ui.cal_factors_plot.addItem(self.cal_factor_plot_data_item)

        self.ui.graphics_widget.nextRow()

        self.ui.tds_voltage_plot = self.ui.graphics_widget.addPlot(title="TDS Amplitude-Voltage Mapping")
        self.ui.tds_voltage_plot.setLabel('bottom', 'TDS Amplitude')
        self.ui.tds_voltage_plot.setLabel('left', 'Voltage / MV')
        self.tds_voltage_plot_data_item = pg.PlotDataItem()
        self.ui.tds_voltage_plot.addItem(self.tds_voltage_plot_data_item)

        self.ui.tds_voltage_plot.setXLink(self.ui.cal_factors_plot)

    def update_plots_and_voltage(self, item):
        if self.updating_table:
            return
        self.updating_table = True

        for row in range(self.ui.table_widget.rowCount()):
            try:
                calibration_item = self.ui.table_widget.item(row, 1)
                cal_factor = float(calibration_item.text()) / CompleteCalibration.CAL_UM_PER_PS
                r = self.ui.r_3412_spin_box.value()
                energy = self.ui.beam_energy_spin_box.value()
                frequency_ghz = self.ui.tds_frequency_spin_box.value()
                frequency_hz = frequency_ghz * 1e9
                voltage_v = calculate_voltage(slope=cal_factor, r34=r, energy=energy, frequency=frequency_hz)
            except (TypeError, ZeroDivisionError, AttributeError, ValueError):
                self._set_voltage_row(row, None)
            else:
                self._set_voltage_row(row, voltage_v)
                # voltage_mv = voltage_v * 1e-6
                # self.ui.table_widget.item(row, 2).setText(f"{voltage_mv:.4g}")

        self.update_plots()
        self.updating_table = False

    def apply_calibration(self):
        self.avmapping_signal.emit(self.get_amplitude_voltage_mapping())

    def _set_cal_factor_row(self, irow, cal_factor_m_per_s):
        calfactor = cal_factor_m_per_s * CompleteCalibration.CAL_UM_PER_PS
        self.ui.table_widget.setItem(irow, 1, QTableWidgetItem(f"{calfactor:.4g}"))

    def _set_voltage_row(self, irow, voltage_v):
        if voltage_v is None:
            try:
                self.ui.table_widget.item(irow, 2).setText("")
            except AttributeError:
                pass
        else:
            voltage_mv = voltage_v * 1e-6
            self.ui.table_widget.item(irow, 2).setText(f"{voltage_mv:.4g}")

    def _get_voltage_row(self, irow):
        voltage_item = self.ui.table_widget.item(irow, 2)
        return float(voltage_item.text()) * 1e6

    def _get_amplitude_row(self, irow):
        amplitude_item = self.ui.table_widget.item(irow, 0)
        return float(amplitude_item.text())

    def update_plots(self):
        amplitudes = []
        calibrations = []
        voltages = []
        for row in range(self.ui.table_widget.rowCount()):
            try:
                amplitude = float(self.ui.table_widget.item(row, 0).text())
                calibration = float(self.ui.table_widget.item(row, 1).text())
                # amplitude.append(float(self.ui.table_widget.item(row, 0).text()))
                # calibration.append(float(self.ui.table_widget.item(row, 1).text()))
                voltage = float(self.ui.table_widget.item(row, 2).text())
                # voltage.append(float(self.ui.table_widget.item(row, 2).text()))
            except (TypeError, ValueError, AttributeError):
                continue
            else:
                amplitudes.append(amplitude)
                calibrations.append(calibration)
                voltages.append(voltage)

        self.cal_factor_plot_data_item.setData(amplitudes, calibrations)
        self.tds_voltage_plot_data_item.setData(amplitudes, voltages)

    def save_table_data(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Table Data", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                # Write headers
                headers = [self.ui.table_widget.horizontalHeaderItem(i).text() for i in range(self.ui.table_widget.columnCount())]
                writer.writerow(headers)
                # Write data rows
                for row in range(self.ui.table_widget.rowCount()):
                    row_data = []
                    for column in range(self.ui.table_widget.columnCount()):
                        item = self.ui.table_widget.item(row, column)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)


def start_bolko_tool():
    app = QtWidgets.QApplication(sys.argv)
    main_window = CalibrationMainWindow()
    main_window.show()
    sys.exit(app.exec_())


def start_calibration_explorer_gui(fname):
    app = QtWidgets.QApplication(sys.argv)
    main_window = CalibrationExplorer(fname)
    main_window.show()
    sys.exit(app.exec_())


def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = CalibrationMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
