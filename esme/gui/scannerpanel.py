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
from esme.gui.common import is_in_controlroom, load_scanner_panel_ui_defaults
from esme.calibration import TDSCalibration

# from esme.maths import ValueWithErrorT, linear_fit

from esme.plot import pretty_parameter_table, formatted_parameter_dfs

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)

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
    tds_amplitude_wait: int = 1.5
    beam_on_wait: float = 1.0
    outdir: Path = Path("./esme-measurements")
    pixel_size: float = 13.7369e-6


class ScannerControl(QtWidgets.QWidget):
    processed_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(FittedBeamParameters)
    background_image_signal = pyqtSignal(object)
    new_measurement_signal = pyqtSignal()

    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        self.ui = Ui_scanner_form()
        self.ui.setupUi(self)

        if machine is None:
            self.machine = build_default_lps_machine()
        else:
            self.machine = machine

        dic = load_scanner_panel_ui_defaults()
        self.initial_read(dic)
        self.settings_dialog = ScannerConfDialog(defaults=dic, parent=self)
        self.connect_buttons()

        self.result_dialog = ScannerResultsDialog(parent=self)
        self.measurement_worker = None
        self.measurement_thread = None

        self.timer = self.build_main_timer(100)

    def load_ui_initial_values(self, dic):
        self.ui.beam_shots_spinner.setValue(dic["nbeam"])
        self.ui.bg_shots_spinner.setValue(dic["nbg"])

    def apply_current_optics(self):
        selected_dispersion = self.get_chosen_dispersion()
        # selected_beta = self.get_chosen_beta()
        setpoint = self.machine.scanner.get_setpoint(selected_dispersion)
        self.machine.scanner.set_scan_setpoint_quads(setpoint)

    def fill_combo_boxes(self):
        self.ui.dispersion_setpoint_combo_box.clear()
        scan = self.machine.scanner.scan
        name = scan.name
        dispersions = [str(s.dispersion) for s in scan.qscan.setpoints]
        self.ui.dispersion_setpoint_combo_box.addItems(dispersions)

        chosen_dispersion = float(self.get_chosen_dispersion())
        chosen_dispersion_setpoint = self.machine.scanner.get_setpoint(chosen_dispersion)

        beta_scan_dispersions = list(set(scan.bscan.dispersions))
        assert len(beta_scan_dispersions) == 1
        beta_scan_dispersion = beta_scan_dispersions[0]
        if beta_scan_dispersion == chosen_dispersion:
            betas = [str(b) for b in scan.bscan.betas]
        else:
            betas = [chosen_dispersion_setpoint.beta]

        self.ui.beta_setpoint_combo_box.addItems(betas)

    # def pick_dispersion(self):
    #     self.ui.beta_setpoint_combo_box.clear()
    #     beta_scan_dispersions = list(set(scan.bscan.dispersions))

    #     chosen_dispersion = float(self.get_chosen_dispersion())
    #     chosen_dispersion_setpoint = self.machine.scanner.get_setpoint(chosen_dispersion)

    def get_chosen_dispersion(self):
        return float(self.ui.dispersion_setpoint_combo_box.currentText())

    # def get_chosen_beta(self):
    #     return float(self.ui.betas_setpoint_combo_box.currentText())

    def open_settings(self):
        self.settings_dialog.show()

    def initial_read(self, dic):

        self.load_ui_initial_values(dic)
        self.fill_combo_boxes()
        voltages = np.array(self.machine.scanner.scan.tscan.voltages)
        voltages /= 1e6
        vstring = ", ".join([str(x) for x in voltages])
        self.ui.tds_voltages.setText(vstring)
        # filling ths stuff
        dscan_tds_voltage_v = self.machine.scanner.scan.qscan.voltage
        dscan_tds_voltage_mv = dscan_tds_voltage_v / 1e6
        self.ui.dispersion_scan_tds_voltage_spinbox.setValue(dscan_tds_voltage_mv)

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
        thread = QThread()
        scan_request = self.build_scan_request_from_ui()
        worker = ScanWorker(self.machine, scan_request)

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
        self.ui.preferences_button.clicked.connect(self.open_settings)
        self.ui.apply_optics_button.clicked.connect(self.apply_current_optics)
        self.ui.start_measurement_button.clicked.connect(self.do_the_measurement)

    def closeEvent(self, event):
        self.measurement_worker.kill = True
        self.measurement_thread.terminate()
        self.measurement_thread.wait()
        # self.screen_thread.exit()

    def display_final_result(self, fitted_beam_parameters):
        LOG.debug("Displaying Final Result")
        try:
            fit_df, beam_df = formatted_parameter_dfs(fitted_beam_parameters)
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("Unable to extract beam parameters from scan(s).")
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            text = pretty_parameter_table(fit_df, beam_df)
            self.result_dialog.ui.result_text_browser.setText(text)
            self.result_dialog.show()
            
        self.measurement_thread.terminate()
        self.measurement_thread.wait()

    def build_scan_request_from_ui(self):
        # XXX: THIS NEEDS TO BE DYNAMIC
        # fname = "/Users/stuartwalker/.config/diagnostics-utility/i1-tds-calibrations/stuart-conf.toml"
        fname = "/System/Volumes/Data/home/xfeloper/user/stwalker/stuart-conf.toml"
        calibration = load_calibration(fname)
        do_beta_scan = self.ui.do_beta_scan_checkbox.isChecked()
        voltages = self.get_voltages()
        slug = self.ui.slug_line_edit.text()
        dscan_tds_voltage_mv = self.ui.dispersion_scan_tds_voltage_spinbox.value()
        dscan_tds_voltage_v = dscan_tds_voltage_mv * 1e6
        images_per_setpoint = self.ui.beam_shots_spinner.value()
        total_background_images = self.ui.bg_shots_spinner.value()
        screen_name = self.machine.scanner.scan.screen

        use_known_resolution = self.ui.use_known_resolution_checkbox.isChecked()

        settings = self.settings_dialog.get_scan_settings()
        # settings = make_default_scan_settings()

        scan_request = ScanRequest(calibration=calibration,
                           voltages=voltages,
                           do_beta_scan=do_beta_scan,
                           dscan_tds_voltage=dscan_tds_voltage_v,
                           screen_name=screen_name,
                           slug=slug,
                           use_known_resolution=use_known_resolution,
                           images_per_setpoint=images_per_setpoint,
                           total_background_images=total_background_images,
                           settings=settings)
        LOG.info(f"Preparing scan request payload: {scan_request}")
        return scan_request


@dataclass
class ScanRequest:
    calibration: TDSCalibration
    voltages: list[float]
    do_beta_scan: bool
    dscan_tds_voltage: float
    screen_name: str
    slug: str
    images_per_setpoint: int
    total_background_images: int
    use_known_resolution: bool
    settings: ScanSettings = None

class ScanWorker(QObject):
    dispersion_sp_signal = pyqtSignal(float)
    processed_image_signal = pyqtSignal(object)
    background_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(FittedBeamParameters)

    def __init__(self, machine, scan_request):
        super().__init__()
        self.machine = machine
        self.scan_request = scan_request
        self.kill = False
        self.output_directory = None

    def get_image_raw_address(self):
        return self.machine.screens.get_image_raw_address(self.scan_request.screen_name)

    def make_output_directory(self):
        basedir = self.scan_request.settings.outdir
        measurement_dir = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%SUTC")
        if self.scan_request.slug:
            measurement_dir = f"{self.scan_request.slug}" + measurement_dir
        directory = Path(basedir) / measurement_dir
        LOG.debug(f"Making output directory: {directory}")
        directory.mkdir(exist_ok=True, parents=True)
        self.output_directory = directory

    def run(self):
        self.make_output_directory()

        background = self.take_background(self.scan_request.total_background_images)

        self.machine.beam_on()
        time.sleep(2)

        bscan_widths = None
        if self.scan_request.do_beta_scan:
            time.sleep(5)
            bscan_widths = self.beta_scan(background)
        time.sleep(5)
        dscan_widths = self.dispersion_scan(background)
        time.sleep(5)
        tscan_widths = self.tds_scan(background)


        fitter = SliceWidthsFitter(dscan_widths=dscan_widths,
                                   tscan_widths=tscan_widths,
                                   bscan_widths=bscan_widths)
        ofp = self.machine.scanner.scan.optics_fixed_points
        print(ofp)
        print("ASSUMING 130MeV BEAM!!!")
        # TODO: beam energy needs to be dynamic!
        # SO DOES VOLTAGE!!!!!!!!!!!!!!
        dscan_voltage = self.scan_request.dscan_tds_voltage
        tscan_dispersion = self.machine.scanner.scan.tscan.setpoint.dispersion
        measurement_result = fitter.all_fit_parameters(beam_energy=130e6,
                                                       dscan_voltage=dscan_voltage,
                                                       tscan_dispersion=tscan_dispersion,
                                                       optics_fixed_points=ofp)
        if self.scan_request.use_known_resolution:
            known_sigma_r = 28e-6
            LOG.info(f"Using known resolution: {known_sigma_r} for calculation")
            fitter.known_sigma_r = known_sigma_r # XXX: MAKE THIS DYNAMIC!!!

        self.full_measurement_result_signal.emit(measurement_result)
        return measurement_result

    def take_background(self, n):
        LOG.info(f"Taking {n} background shots...")
        self.machine.beam_off()
        time.sleep(2)
        bgs = []
        screen_name = self.scan_request.screen_name
        for _ in range(5):
            time.sleep(0.2)
            raw_image = self.machine.screens.get_image_raw(screen_name)
            bgs.append(raw_image.T)
            self.background_image_signal.emit(raw_image)

        mean_bg = np.mean(bgs, axis=0)
        return mean_bg

    def dispersion_scan(self, bg) -> dict[float, float]:
        voltage = self.scan_request.dscan_tds_voltage
        # voltage =
        self.set_tds_voltage(voltage)
        print("Doing dispersion scan at voltage", voltage)
        # print("Doing dispersion scan at amplitude", amplitude)
        widths = defaultdict(list)
        for setpoint in self.machine.scanner.scan.qscan.setpoints:
            self.set_quads(setpoint)
            time.sleep(self.scan_request.settings.quad_wait)
            sp_widths = self.do_one_scan_setpoint(ScanType.DISPERSION,
                                                  dispersion=setpoint.dispersion,
                                                  voltage=voltage,
                                                  beta=setpoint.beta,
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
        for i, voltage in enumerate(self.scan_request.voltages):
            self.set_tds_voltage(voltage)
            # time.sleep(self.scan_settings.tds_amplitude_wait)
            if i == 0:
                time.sleep(3)
            time.sleep(5)
            sp_widths = self.do_one_scan_setpoint(ScanType.TDS,
                                                  dispersion=setpoint.dispersion,
                                                  voltage=voltage,
                                                  beta=setpoint.beta,
                                                  bg=bg)
            widths[voltage].extend(sp_widths)

        widths = {voltage: np.mean(widths) for voltage, widths in widths.items()}
        return widths

    def beta_scan(self, bg) -> dict[float, float]:
        voltage = self.scan_request.dscan_tds_voltage
        self.set_tds_voltage(voltage)
        print("Doing beta scan at voltage", voltage)
        widths = defaultdict(list)

        for setpoint in self.machine.scanner.scan.bscan.setpoints:
            print(f"Starting beta scan setpoint={setpoint.beta}.")
            self.set_quads(setpoint)
            time.sleep(self.scan_request.settings.quad_wait)
            sp_widths = self.do_one_scan_setpoint(ScanType.BETA,
                                                  setpoint.dispersion,
                                                  voltage=voltage,
                                                  beta=setpoint.beta,
                                                  bg=bg)
            widths[setpoint.beta].extend(sp_widths)
        # Get the average...
        widths = {dx: np.mean(widths) for dx, widths in widths.items()}
        return widths

    def do_one_scan_setpoint(self, scan_type: ScanType, dispersion: float, voltage: float, beta: float, bg=0.0):
        widths = [] # Result
        # Output pandas dataframe of snapshots

        with self.snapshot_accumulator(scan_type, dispersion, voltage) as accumulator:
            for raw_image in self.take_beam_data(self.scan_request.images_per_setpoint):
                processed_image = process_image(raw_image, scan_type,
                                                dispersion=dispersion,
                                                voltage=voltage,
                                                beta=beta,
                                                bg=bg)
                accumulator.take_snapshot(raw_image,
                                          dispersion=dispersion,
                                          voltage=voltage,
                                          beta=beta,
                                          scan_type=str(scan_type))
                widths.append(processed_image.central_width)
                self.processed_image_signal.emit(processed_image)
            return widths

    def set_quads(self, setpoint):
        self.machine.scanner.set_scan_setpoint_quads(setpoint)
        self.dispersion_sp_signal.emit(setpoint.dispersion)

    def set_tds_voltage(self, voltage):
        LOG.info(f"Setting TDS voltage: {voltage / 1e6} MV")
        amplitude = self.scan_request.calibration.get_amplitude(voltage)
        self.machine.deflectors.active_tds().set_amplitude(amplitude)

    def take_beam_data(self, nbeam):
        screen_name = self.scan_request.screen_name
        for _ in range(nbeam):
            time.sleep(0.2)
            raw_image = self.machine.screens.get_image_raw(screen_name)
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

def process_image(image, scan_type: ScanType, dispersion, voltage, beta, bg=0):
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
                          voltage=voltage,
                          beta=beta)



class ScannerConfDialog(QtWidgets.QDialog):
    scanner_config_signal = pyqtSignal(ScanSettings)
    def __init__(self, defaults, parent=None):
        super().__init__()
        self.ui = scanner_config.Ui_Dialog()
        self.ui.setupUi(self)
        self.defaults = defaults
        self.restore_ui_initial_values()

    def restore_ui_initial_values(self):
        uiconf = self.defaults
        quad_wait = uiconf["quad_wait"]
        # nbg = uiconf["nbackground_per_scan"]
        # nbeam = uiconf["nbeam_per_setpoint"]
        quad_wait = uiconf["quad_wait"]
        tds_amplitude_wait = uiconf["tds_amplitude_wait"]
        beam_on_wait = uiconf["beam_on_wait"]
        self.ui.tds_amplitude_wait_spinbox.setValue(tds_amplitude_wait)
        self.ui.quad_sleep_spinbox.setValue(quad_wait)
        self.ui.beam_on_wait_spinbox.setValue(beam_on_wait)

        if is_in_controlroom():
            outdir = Path(uiconf["outdir_bkr"])
        else:
            outdir = Path("./measurements")

        outdir = str(Path(outdir.resolve()))

        self.ui.output_directory_lineedit.setText(outdir)

    def get_scan_settings(self):
        return ScanSettings(quad_wait=self.ui.quad_sleep_spinbox.value(),
                            tds_amplitude_wait=self.ui.tds_amplitude_wait_spinbox.value(),
                            beam_on_wait=self.ui.beam_on_wait_spinbox.value(),
                            outdir=self.ui.output_directory_lineedit.text())


@dataclass
class NewSetpointSignal:
    voltage: float
    dispersion: float
    beta: float

class ScannerResultsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_results_box_dialog()
        self.ui.setupUi(self)

        self.ui.send_to_logbook_button.clicked.connect(self.send_to_logbook)

    def post_result(self, fitted_beam_parameters):
        pass

    def send_to_logbook(self):
        text = self.ui.result_text_browser.toPlainText()
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
    beta: float





def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = ScannerControl()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
