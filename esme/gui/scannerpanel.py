from datetime import datetime
from pathlib import Path
import time
from dataclasses import dataclass
import sys
import logging
from collections import defaultdict
from enum import Enum, auto

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox

from esme.gui.ui.scanner import Ui_scanner_form
from esme.gui.ui.scan_results import Ui_results_box_dialog
from esme.gui.ui import scanner_config
from esme.control.pattern import get_beam_regions, get_bunch_pattern
from esme.gui.common import build_default_lps_machine, QPlainTextEditLogger
from esme.maths import ValueWithErrorT, linear_fit
from esme.image import get_slice_properties, get_central_slice_width_from_slice_properties, filter_image
from esme.analysis import SliceWidthsFitter, FittedBeamParameters
from esme.control.configs import load_calibration
from esme.control.snapshot import SnapshotAccumulator, Snapshotter
from esme.control.mint import send_to_logbook
from esme.gui.common import is_in_controlroom
# from esme.maths import ValueWithErrorT, linear_fit

from esme.plot import pretty_parameter_table, formatted_parameter_dfs

LOG = logging.getLogger(__name__)

# A Coursera course on algorithms & data structures and another one on machine learning will be useful

# Once I felt I was ready to face some interviews I started to do a lot
# of them (1 to 3 coding interviews per week). After 2 months many
#  successful and failed interviews (there are several rounds per
#                                    position) I managed to get two offers, one for a cool but modest company in Cambridge, and other in Goldman Sachs, so I decided to go for GS.


class ScanType(Enum):
    DISPERSION = auto()
    TDS = auto()
    BETA = auto()

    @classmethod
    @property
    def ALT_NAME_MAP(cls):
        return {cls.DISPERSION: "dscan", cls.TDS: "tscan", cls.BETA: "bscan"}

    def alt_name(self):
        return self.ALT_NAME_MAP[self]

@dataclass
class ScanSettings:
    quad_wait: float = 5
    nbeam: int = 20
    tds_amplitude_wait: int = 1.5
    beam_on_wait: float = 1.0
    outdir: Path = Path("/Users/xfeloper/user/stwalker/esme-measurements")
    pixel_size: float = 13.7369e-6


def make_default_scan_settings():
    ss = ScanSettings()
    if is_in_controlroom():
        ss.outdir = "/Users/xfeloper/user/stwalker/esme-measurements"
    return ss

class ScannerControl(QtWidgets.QWidget):
    processed_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(FittedBeamParameters)
    background_image_signal = pyqtSignal(object)
    new_measurement_signal = pyqtSignal()

    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        self.ui = Ui_scanner_form()
        self.ui.setupUi(self)

        self.scan_settings = make_default_scan_settings()

        if machine is None:
            self.machine = build_default_lps_machine()
        else:
            self.machine = machine

        self.initial_read()
        self.connect_buttons()

        self.measurement_worker = None
        self.measurement_thread = None

        self.timer = self.build_main_timer(100)

        self.result_dialog = ScannerResultsDialog(parent=self)
        self.settings_dialog = ScannerConfDialog(initial_settings=make_default_scan_settings(),
                                                 parent=self)
        self.settings_dialog.scanner_config_signal.connect(self.update_settings)
        self.ui.preferences_button.clicked.connect(self.open_settings)

        self.ui.apply_optics_button.clicked.connect(self.apply_current_optics)

    def apply_current_optics(self):
        selected_dispersion = float(self.ui.dispersion_setpoint_combo_box.currentText())
        setpoint = self.machine.scanner.get_setpoint(selected_dispersion)
        self.machine.scanner.set_scan_setpoint_quads(setpoint)

    def fill_combo_boxes(self):
        self.ui.dispersion_setpoint_combo_box.clear()
        scan = self.machine.scanner.scan
        name = scan.name
        dispersions = [str(s.dispersion) for s in scan.qscan.setpoints]
        self.ui.dispersion_setpoint_combo_box.addItems(dispersions)
        

    def update_settings(self):
        pass

    def open_settings(self):
        self.settings_dialog.show()

    def initial_read(self):
        self.fill_combo_boxes()
        voltages = np.array(self.machine.scanner.scan.tscan.voltages)
        voltages /= 1e6
        vstring = ", ".join([str(x) for x in voltages])
        self.ui.tds_voltages.setText(vstring)
        # filling ths stuff
        dscan_tds_voltage = self.machine.scanner.scan.qscan.voltage
        self.ui.dispersion_scan_tds_voltage_spinbox.setValue(dscan_tds_voltage)
        self.ui.beam_shots_spinner.setValue(self.scan_settings.nbeam)

        # voltages = np.array(self.machine.scanner.scan.qscan.voltages)        
        # self.ui.dispersion_scan_tds_voltage_spinbox.setValue(

    def get_voltages(self):
        vstring = self.ui.tds_voltages.text()
        numbers = vstring.split(",")
        voltages = np.array([float(n) for n in numbers])
        voltages *= 1e6
        return voltages

    def build_main_timer(self, period):
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        # timer.timeout.connect(self.update)
        timer.start(period)
        return timer

    def do_the_measurement(self):
        fname = "/System/Volumes/Data/home/xfeloper/user/stwalker/stuart-conf.toml"
        calibration = load_calibration(fname)
        thread = QThread()
        worker = ScanWorker(self.machine, calibration, self.get_voltages(),
                            slug=self.ui.slug_line_edit.text())
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.processed_image_signal.connect(self.processed_image_signal.emit)
        worker.background_image_signal.connect(self.background_image_signal.emit)        
        worker.full_measurement_result_signal.connect(self.full_measurement_result_signal.emit)
        worker.full_measurement_result_signal.connect(self.display_final_result)
        thread.start()
        self.measurement_thread = thread
        self.measurement_worker = worker

    def connect_buttons(self):
        self.ui.start_measurement_button.clicked.connect(self.do_the_measurement)

    def closeEvent(self, event):
        self.measurement_worker.kill = True
        self.measurement_thread.terminate()
        self.measurement_thread.wait()
        # self.screen_thread.exit()

    def display_final_result(self, fitted_beam_parameters):
        LOG.debug("Displaying Final Result")
        fit_df, beam_df = formatted_parameter_dfs(fitted_beam_parameters)
        text = pretty_parameter_table(fit_df, beam_df)
        self.result_dialog.ui.result_text_browser.setText(text)
        self.result_dialog.show()
        self.measurement_thread.terminate()
        self.measurement_thread.wait()


class ScanWorker(QObject):
    dispersion_sp_signal = pyqtSignal(float)
    processed_image_signal = pyqtSignal(object)
    background_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(FittedBeamParameters)

    def __init__(self, machine, calibration, voltages, slug=None):
        super().__init__()
        self.machine = machine
        self.scan_settings = ScanSettings()
        self.calibration = calibration
        self.voltages = voltages
        self.screen_name = "OTRC.64.I1D"
        self.slug = slug
        self.kill = False
        self.output_directory = None

    def get_image_raw_address(self):
        return self.machine.screens.get_image_raw_address(self.screen_name)

    def make_output_directory(self):
        basedir = self.scan_settings.outdir
        measurement_dir = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%SUTC")
        if self.slug:
            measurement_dir = f"{self.slug}" + measurement_dir
        directory = basedir / measurement_dir
        LOG.debug(f"Making output directory: {directory}")
        directory.mkdir(exist_ok=True, parents=False)
        self.output_directory = directory

    def run(self):
        self.make_output_directory()

        background = self.take_background()

        self.machine.beam_on()
        time.sleep(2)

        dscan_widths = self.dispersion_scan(background)
        time.sleep(5)
        tscan_widths = self.tds_scan(background)


        fitter = SliceWidthsFitter(dscan_widths,
                                   tscan_widths,
                                   # self.machine.scanner.scan
                                   )
        ofp = self.machine.scanner.scan.optics_fixed_points

        # TODO: beam energy needs to be dynamic!
        # SO DOES VOLTAGE!!!!!!!!!!!!!!
        dscan_voltage = self.machine.scanner.scan.qscan.voltage
        tscan_dispersion = self.machine.scanner.scan.tscan.setpoint.dispersion
        measurement_result = fitter.all_fit_parameters(beam_energy=130e6,
                                                       dscan_voltage=dscan_voltage,
                                                       tscan_dispersion=tscan_dispersion,
                                                       optics_fixed_points=ofp)
        self.full_measurement_result_signal.emit(measurement_result)
        return measurement_result

    def take_background(self):
        self.machine.beam_off()
        time.sleep(2)
        bgs = []
        for _ in range(5):
            time.sleep(0.2)
            raw_image = self.machine.screens.get_image_raw(self.screen_name)
            bgs.append(raw_image.T)
            self.background_image_signal.emit(raw_image)

        mean_bg = np.mean(bgs, axis=0)
        print(mean_bg)
        print(mean_bg.shape)
        return mean_bg

    def dispersion_scan(self, bg) -> dict[float, float]:
        voltage = self.machine.scanner.scan.qscan.voltage
        voltage = 0.61e6
        # voltage = 
        self.set_tds_voltage(voltage)
        print("Doing dispersion scan at voltage", voltage)
        # print("Doing dispersion scan at amplitude", amplitude)        
        widths = defaultdict(list)
        for setpoint in self.machine.scanner.scan.qscan.setpoints:
            self.set_quads(setpoint)
            time.sleep(self.scan_settings.quad_wait)
            sp_widths = self.do_one_scan_setpoint(ScanType.DISPERSION,
                                               setpoint.dispersion,
                                                  voltage,
                                                  bg=bg)
            widths[setpoint.dispersion].extend(sp_widths)
        # Get the average...
        widths = {dx: np.mean(widths) for dx, widths in widths.items()}
        return widths

    def tds_scan(self, bg) -> dict[float, float]:
        setpoint = self.machine.scanner.scan.tscan.setpoint
        print("Doing tds scan at dispersion", setpoint.dispersion)
        self.set_quads(setpoint)
        time.sleep(3)
        widths = defaultdict(list)
        for i, voltage in enumerate(self.voltages):
            self.set_tds_voltage(voltage)
            # time.sleep(self.scan_settings.tds_amplitude_wait)
            if i == 0:
                time.sleep(3)
            time.sleep(5)
            sp_widths = self.do_one_scan_setpoint(ScanType.TDS,
                                                  setpoint.dispersion,
                                                  voltage,
                                                  bg=bg)
            widths[voltage].extend(sp_widths)

        widths = {voltage: np.mean(widths) for voltage, widths in widths.items()}
        return widths

    def do_one_scan_setpoint(self, scan_type: ScanType, dispersion: float, voltage: float, bg=0.0):
        widths = [] # Result
        # Output pandas dataframe of snapshots

        with self.snapshot_accumulator(scan_type, dispersion, voltage) as accumulator:
            for raw_image in self.take_beam_data(self.scan_settings.nbeam):
                processed_image = process_image(raw_image, scan_type,
                                                dispersion=dispersion,
                                                voltage=voltage,
                                                bg=bg)
                accumulator.take_snapshot(raw_image,
                                          dispersion=dispersion,
                                          voltage=voltage,
                                          scan_type=str(scan_type))
                widths.append(processed_image.central_width)
                self.processed_image_signal.emit(processed_image)
            return widths

    def set_quads(self, setpoint):
        self.machine.scanner.set_scan_setpoint_quads(setpoint)
        self.dispersion_sp_signal.emit(setpoint.dispersion)

    def set_tds_voltage(self, voltage):
        LOG.info(f"Setting TDS voltage: {voltage / 1e6} MV")
        amplitude = self.calibration.get_amplitude(voltage)
        self.machine.deflectors.active_tds().set_amplitude(amplitude)

    def take_beam_data(self, nbeam):
        for _ in range(nbeam):
            time.sleep(0.2)
            raw_image = self.machine.screens.get_image_raw(self.screen_name)
            yield raw_image

    def snapshot_accumulator(self, scan_type, dispersion, voltage):
        shotter = self.machine.scanner.get_snapshotter()
        image_address = self.get_image_raw_address()
        outdir = self.output_directory
        filename = make_snapshot_filename(scan_type=scan_type, dispersion=dispersion, voltage=voltage)
        return SnapshotAccumulator(shotter, outdir / filename)



def make_snapshot_filename(*, scan_type, dispersion, voltage, **images):
    scan_string = scan_type.alt_name()
    voltage /= 1e6
    return f"{scan_string}-V={voltage=}MV_{dispersion=}m.npz"

def process_image(image, scan_type: ScanType, dispersion, voltage, bg=0):
    image = image.T # Flip to match control room..?  TODO
    image = filter_image(image, bg=bg, crop=True)
    _, means, sigmas = get_slice_properties(image)
    sigma = get_central_slice_width_from_slice_properties(
        means, sigmas, padding=10
    )
    central_width_row = np.argmin(means)
    return ProcessedImage(image, scan_type,
                          central_width=sigma,
                          central_width_row=central_width_row,
                          dispersion=dispersion,
                          voltage=voltage)



class ScannerConfDialog(QtWidgets.QDialog):
    scanner_config_signal = pyqtSignal(ScanSettings)
    def __init__(self, initial_settings=None, parent=None):
        super().__init__()
        self.ui = scanner_config.Ui_Dialog()
        self.ui.setupUi(self)

        self.update_settings(initial_settings)

    def update_settings(self, initial_settings):
        if not initial_settings:
            return

        self.ui.tds_amplitude_wait_spinbox.setValue(initial_settings.tds_amplitude_wait)
        self.ui.quad_sleep_spinbox.setValue(initial_settings.quad_wait)
        self.ui.output_directory_lineedit.setText(str(initial_settings.outdir))
        self.ui.pixel_size_spinbox.setValue(initial_settings.pixel_size)
        self.ui.beam_on_wait_spinbox.setValue(initial_settings.beam_on_wait)

    def get_scan_settings(self):
        return ScanSettings(quad_wait=self.ui.tds_amplitude_wait_spinbox.value(),
                            tds_amplitude_wait=self.ui.tds_amplitude_wait_spinbox.value(),
                            beam_on_wait=self.ui.beam_on_wait_spinbox.value(),
                            outdir=self.ui.output_directory_lineedit.text())


class ScannerResultsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_results_box_dialog()
        self.ui.setupUi(self)

        self.ui.send_to_logbook_button.clicked.connect(self.send_to_logbook)

    def post_result(self, fitted_beam_parameters):
        pass

    def send_to_logbook(self):
        #?????????? I don't know how to get this to work...
        # pixmap = self.parent().parent().parent().grab()
        # size = pixmap.size()
        # h = size.width()
        # w = size.height()

        # image = pixmap.toImage()
        # byte_str = image.bits().tobytes()
        # img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))

        text = self.ui.result_text_browser.text()
        send_to_logbook(title="Slice Energy Spread Measurement",
                        author="WAL",
                        text=text)




# class ResultDisplayBox:

@dataclass
class ProcessedImage:
    image: np.ndarray
    scan_type: ScanType
    central_width: tuple
    central_width_row: int
    dispersion: float
    voltage: float





def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = ScannerControl()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
