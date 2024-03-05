from datetime import datetime
from pathlib import Path
import time
from dataclasses import dataclass
import sys
import logging
from collections import defaultdict
from enum import Enum, auto
from copy import deepcopy

import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from scipy.constants import e
from uncertainties import UFloat


from esme.gui.ui import Ui_scanner_form, Ui_Dialog
from esme.image import filter_image, get_selected_central_slice_width_from_slice_properties, get_slice_properties
from esme.analysis import SliceWidthsFitter, DerivedBeamParameters, true_bunch_length_from_processed_image, ScanType
from esme.control.snapshot import SnapshotAccumulator
from esme.gui.common import (is_in_controlroom,
                             load_scanner_panel_ui_defaults,
                             raise_message_box,
                             make_default_injector_espread_machine)
from esme.calibration import AmplitudeVoltageMapping
from esme.optics import load_matthias_slice_measurement


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


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


@dataclass
class OnlineMeasurementResult:
    measurement_parameters: DerivedBeamParameters
    output_directory: Path


class ScannerControl(QtWidgets.QWidget):
    processed_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(OnlineMeasurementResult)
    background_image_signal = pyqtSignal(object)
    new_measurement_signal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.ui = Ui_scanner_form()
        self.ui.setupUi(self)

        self.machine = make_default_injector_espread_machine()

        ui_defaults = load_scanner_panel_ui_defaults()
        self.initial_read(ui_defaults["ScannerControl"])
        self.settings_dialog = ScannerConfDialog(defaults=ui_defaults["ScannerConfDialog"],
                                                 parent=self)
        self.measured_slice_twiss = None
        self.connect_buttons()

        self.result_dialog = ScannerResultsDialog(parent=self)
        self.measurement_worker = None
        self.measurement_thread = None
        self.jddd_camera_window_process = None

        self.timer = self.build_main_timer(100)


    def set_ui_initial_values(self, dic):
        self.ui.beam_shots_spinner.setValue(dic["beam_shots_spinner"])
        self.ui.bg_shots_spinner.setValue(dic["bg_shots_spinner"])

    def apply_current_optics(self):
        selected_dispersion = self.get_chosen_dispersion()
        selected_beta = self.get_chosen_beta()
        setpoint = self.machine.scanner.get_setpoint(selected_dispersion,
                                                     beta=selected_beta)
        self.machine.scanner.set_scan_setpoint_quads(setpoint)

    def fill_combo_boxes(self):
        self.ui.dispersion_setpoint_combo_box.clear()
        scan = self.machine.scanner.scan
        scan.name
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

    def initial_read(self, dic):
        self.set_ui_initial_values(dic)
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
        """Parse TDS voltages string from TDS voltage scan box in MV
        and convert to V, returned.

        """
        vstring = self.ui.tds_voltages.text()
        numbers = vstring.split(",")
        voltages = np.array([float(n) for n in numbers])
        voltages *= 1e6
        return voltages

    def build_main_timer(self, period):
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        timer.start(period)
        return timer

    def start_measurement(self):
        thread = QThread()

        try:
            scan_request = self.build_scan_request_from_ui()
        except MisconfiguredMeasurementException as e:
            raise_message_box(text="Error in preparing measurement",
                              informative_text=f"{e}.  The measurement cannot proceed.",
                              title="Measurement Preparation Error",
                              icon="Critical")
            return


        worker = ScanWorker(self.machine, scan_request)

        self.set_buttons_ready_for_measurement(can_measure=False)

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        thread.finished.connect(self.stop_measurement)
        worker.processed_image_signal.connect(self.processed_image_signal.emit)
        worker.background_image_signal.connect(self.background_image_signal.emit)
        worker.full_measurement_result_signal.connect(self.full_measurement_result_signal.emit)
        worker.full_measurement_result_signal.connect(self.display_final_result)
        worker.measurement_interrupted_signal.connect(self.handle_measurement_exception)
        thread.start()
        self.measurement_thread = thread
        self.measurement_worker = worker

    def stop_measurement(self):
        self.kill_measurement_thread()
        self.set_buttons_ready_for_measurement(can_measure=True)

    def set_buttons_ready_for_measurement(self, can_measure):
        """Enable / disable UI features based on whether or not the
        measurement is running already or not

        """
        self.ui.stop_measurement_button.setEnabled(not can_measure)
        self.ui.start_measurement_button.setEnabled(can_measure)
        self.ui.dispersion_setpoint_combo_box.setEnabled(can_measure)
        self.ui.beta_setpoint_combo_box.setEnabled(can_measure)
        self.ui.apply_optics_button.setEnabled(can_measure)
        self.ui.do_beta_scan_checkbox.setEnabled(can_measure)
        self.ui.tds_voltages.setEnabled(can_measure)
        self.ui.dispersion_scan_tds_voltage_spinbox.setEnabled(can_measure)
        self.ui.bg_shots_spinner.setEnabled(can_measure)
        self.ui.beam_shots_spinner.setEnabled(can_measure)
        self.ui.cycle_quads_button.setEnabled(can_measure)
        self.ui.slug_line_edit.setEnabled(can_measure)
        self.ui.background_shots_label.setEnabled(can_measure)
        self.ui.beam_shots_label.setEnabled(can_measure)
        self.ui.beta_label.setEnabled(can_measure)
        self.ui.dispersion_label.setEnabled(can_measure)
        self.ui.tds_voltages_label.setEnabled(can_measure)
        self.ui.dscan_voltage_label.setEnabled(can_measure)
        self.ui.measurement_name_label.setEnabled(can_measure)
        self.ui.preferences_button.setEnabled(can_measure)
        self.ui.load_quad_scan_button.setEnabled(can_measure)
        self.ui.do_full_phase_space_checkbox.setEnabled(can_measure and bool(self.measured_slice_twiss))

    def connect_buttons(self):
        """Connect the buttons of the UI to the relevant methods"""
        self.ui.preferences_button.clicked.connect(self.settings_dialog.show)
        self.ui.apply_optics_button.clicked.connect(self.apply_current_optics)
        self.ui.start_measurement_button.clicked.connect(self.start_measurement)
        self.ui.stop_measurement_button.clicked.connect(self.stop_measurement)
        self.ui.open_jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.cycle_quads_button.clicked.connect(self.machine.scanner.cycle_scan_quads)
        self.ui.show_optics_button.clicked.connect(self.show_optics_at_screen)
        self.ui.load_quad_scan_button.clicked.connect(self.load_quad_scan_file)
        self.set_buttons_ready_for_measurement(can_measure=True)

    def load_quad_scan_file(self):
        initial_directory = "/home/xfeloper/data/quad_scan/2023/"
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load Matthias Slice Emittance Measurement",
            directory=initial_directory,
            filter="mat files (*.mat)",
            initialFilter="mat files",
            options=QFileDialog.Options(),
        )
        self.measured_slice_twiss = load_matthias_slice_measurement(fname)

    def show_optics_at_screen(self):
        fname = "/Users/stuartwalker/Downloads/2023-08-31T221036_quad_scan_slice_I1_h_slice_OTRC_59_I1_screen.mat"
        self.measured_slice_twiss = load_matthias_slice_measurement(fname)

        stwiss0 = self.measured_slice_twiss

        slice_twiss1 = self.machine.optics.track_measured_slice_twiss(stwiss0=stwiss0,
                                                                      start="QI.53.I1",
                                                                      stop="OTRC.64.I1D")

        from esme.gui.mpl_widget import MatplotlibCanvas
        widget = MatplotlibCanvas(parent=self, nrows=2, sharex=True)
        nslices = stwiss0.nslices

        indices = np.arange(-nslices // 2, nslices // 2)

        top = widget.axes[0]

        widget.axes[0].plot(indices, list(slice_twiss1.beta_x), label="Propagated Measurement", linestyle="", marker="x")
        widget.axes[0].axhline(0.6, label="Expected (hardcoded)", linestyle="--")
        widget.axes[1].plot(indices, list(slice_twiss1.alpha_x), linestyle="", marker="x")

        top.legend()
        widget.axes[0].set_title("Slice Twiss Parameters at OTRC.64.I1D")

        widget.axes[1].set_xlabel("Slice Index")
        widget.axes[0].set_ylabel(r"$\beta_x$ / m")
        widget.axes[1].set_ylabel(r"$\alpha_x$")

        widget.show()


    def closeEvent(self, event):
        """called automatically when closed, terminate the daughter
        thread for the measurement.

        """
        self.kill_measurement_thread()

    def display_final_result(self, result: OnlineMeasurementResult):
        LOG.debug("Displaying Final Result")
        try:
            self.result_dialog.post_measurement_result(result)
        except ValueError:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Error")
            msg.setInformativeText("Unable to extract beam patrameters from scan(s).")
            msg.setWindowTitle("Error")
            msg.exec_()
        else:
            self.result_dialog.show()

        self.measurement_thread.terminate()
        self.measurement_thread.wait()
        self.set_buttons_ready_for_measurement(can_measure=True)

    def build_scan_request_from_ui(self):
        calibration = self.machine.deflector.calibration
        if calibration is None:
            raise MisconfiguredMeasurementException("Missing TDS Calibration")

        do_beta_scan = self.ui.do_beta_scan_checkbox.isChecked()
        voltages = self.get_voltages()
        slug = self.ui.slug_line_edit.text()
        dscan_tds_voltage_mv = self.ui.dispersion_scan_tds_voltage_spinbox.value()
        dscan_tds_voltage_v = dscan_tds_voltage_mv * 1e6
        images_per_setpoint = self.ui.beam_shots_spinner.value()
        total_background_images = self.ui.bg_shots_spinner.value()
        screen_name = self.machine.scanner.scan.screen

        if self.ui.do_full_phase_space_checkbox.isChecked():
            self.measured_slice_twiss

        settings = self.settings_dialog.get_scan_settings()

        slice_pos = None
        if self.ui.pick_slice_based_on_position_checkbox.isChecked():
            slice_pos = self.ui.slice_selection_spinner.value()

        scan_request = ScanRequest(calibration=calibration,
                                   voltages=voltages,
                                   do_beta_scan=do_beta_scan,
                                   dscan_tds_voltage=dscan_tds_voltage_v,
                                   screen_name=screen_name,
                                   slug=slug,
                                   slice_pos=slice_pos,
                                   images_per_setpoint=images_per_setpoint,
                                   total_background_images=total_background_images,

                                   settings=settings)
        LOG.info(f"Preparing scan request payload: {scan_request}")
        return scan_request

    def update_tds_calibration(self, calibration):
        print(calibration, calibration.region)

        # XXX: what if I receive a TDS calibration from I1 but I am
        # applying it here to B2?  This would obviously be wrong...
        # Need to be careful here and think about this in the future.
        self.machine.deflector.calibration = calibration

    def open_jddd_screen_window(self):
        self.jddd_camera_window_process = QtCore.QProcess()
        screen = self.machine.scanner.scan.screen
        LOG.info(f"Opening JDDD Screen Window: {screen}")
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen}/"
        LOG.debug("Calling %s", command)
        self.jddd_camera_window_process.start(command)
        self.jddd_camera_window_process.waitForStarted()
        self.jddd_camera_window_process.finished.connect(self.close_jddd_screen_window)

    def close_jddd_screen_window(self):
        self.jddd_camera_window_process.close()
        self.jddd_camera_window_process = None

    def handle_measurement_exception(self, exception: Exception):
        if isinstance(exception, UserCancelledMeasurementException):
            return
        elif isinstance(exception, MachineCancelledMeasurementException):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Machine Error")
            msg.setInformativeText(f"Measurement cancelled: {exception}")
            msg.setWindowTitle("Error")
            msg.exec_()
        self.set_buttons_ready_for_measurement(can_measure=True)
        self.kill_measurement_thread()

    def kill_measurement_thread(self):
        self.measurement_worker.kill = True
        self.measurement_thread.terminate()
        self.measurement_thread.wait()


@dataclass
class ScanRequest:
    calibration: AmplitudeVoltageMapping
    voltages: list[float]
    do_beta_scan: bool
    dscan_tds_voltage: float
    screen_name: str
    slug: str
    images_per_setpoint: int
    total_background_images: int
    slice_pos: int = None
    # measured_slice_twiss = None
    settings: ScanSettings = None


@dataclass
class ProcessedImage:
    image: np.ndarray
    scan_type: ScanType
    central_width: tuple
    central_width_row: int
    sigma_z: tuple
    dispersion: float
    voltage: float
    beta: float
    

class ScanWorker(QObject):
    dispersion_sp_signal = pyqtSignal(float)
    processed_image_signal = pyqtSignal(object)
    background_image_signal = pyqtSignal(object)
    full_measurement_result_signal = pyqtSignal(OnlineMeasurementResult)
    measurement_interrupted_signal = pyqtSignal(Exception)

    def __init__(self, machine, scan_request):
        super().__init__()
        self.machine = machine
        self.scan_request = scan_request
        self.kill = False
        self.output_directory = None

    # def get_image_raw_address(self):
    #     screen =
    #     return self.machine.screens.get_image_raw_address(self.scan_request.screen_name)

    def make_output_directory(self):
        basedir = self.scan_request.settings.outdir
        measurement_dir = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%SUTC")
        if self.scan_request.slug:
            measurement_dir = f"{self.scan_request.slug}" + measurement_dir
        directory = Path(basedir) / measurement_dir
        LOG.debug(f"Making output directory: {directory}")
        directory.mkdir(exist_ok=True, parents=True)
        self.output_directory = directory

    def setup_special_bunch_midlayer(self):
        """Set up the special bunch midlayer for this measurement."""
        sbml = self.machine.sbunches
        sbml.set_npulses(1_000_000) # A big number...
        # We only have one beam region and one bunch with a screen in:
        sbml.set_bunch_number(1)
        sbml.set_beam_region(0)
        # Of course we use the TDS
        sbml.set_use_tds(use_tds=True)
        # Kickers I think have to be powered on to make the SBM server
        # happy, but obivously we don't use them.
        sbml.power_on_kickers()
        sbml.dont_use_kickers()

    def start_tds_firing(self):
        self.machine.sbunches.start_diagnostic_bunch()

    def run(self):
        self.make_output_directory()

        try:
            background = self.take_background(self.scan_request.total_background_images)
        except InterruptedMeasurementException as e:
            self.measurement_interrupted_signal.emit(e)
            return

        # Setup the special bunch midlayer, we benefit a lot from
        # using this over directly affecting the TDS timing because we
        # don't have to worry about the blms complaining.
        if is_in_controlroom():
            self.setup_special_bunch_midlayer()

        # Turn the beam on and put the TDS on beam and wait.
        self.machine.beam_on()
        self.start_tds_firing()
        time.sleep(self.scan_request.settings.beam_on_wait)

        try:
            bscan_widths = None
            if self.scan_request.do_beta_scan:
                time.sleep(5)
                bscan_widths = self.beta_scan(background)
            time.sleep(5)
            dscan_widths = self.dispersion_scan(background)
            time.sleep(5)
            tscan_widths, tscan_bunch_lengths = self.tds_scan(background)
        except InterruptedMeasurementException as e:
            self.measurement_interrupted_signal.emit(e)
            return

        # bunch_length = np.mean(tscan_bunch_lengths[max(tscan_bunch_lengths)])
        bunch_length = np.mean(list(tscan_bunch_lengths.values()))


        fitter = SliceWidthsFitter(dscan_widths=dscan_widths,
                                   tscan_widths=tscan_widths,
                                   bscan_widths=bscan_widths)
        ofp = self.machine.scanner.scan.optics_fixed_points
        beam_energy = self.machine.optics.get_beam_energy() * 1e6
        dscan_voltage = self.scan_request.dscan_tds_voltage

        tscan_dispersion = self.machine.scanner.scan.tscan.setpoint.dispersion
        derived_beam_parameters = fitter.all_fit_parameters(beam_energy=beam_energy,
                                                            dscan_voltage=dscan_voltage,
                                                            tscan_dispersion=tscan_dispersion,
                                                            optics_fixed_points=ofp,
                                                            sigma_z=(bunch_length.n, bunch_length.s))

        result = OnlineMeasurementResult(measurement_parameters=derived_beam_parameters,
                                         output_directory=self.output_directory)
        self.full_measurement_result_signal.emit(result)
        return derived_beam_parameters

    def take_background(self, n):
        if self.kill:
            raise InterruptedMeasurementException

        LOG.info(f"Taking {n} background shots...")
        self.machine.beam_off()
        time.sleep(2)

        bgs = []
        with self.background_data_accumulator() as accu:
            for raw_image in self.take_screen_data(n, expect_beam=False):
                if self.kill:
                    raise UserCancelledMeasurementException
                accu.take_snapshot(raw_image,
                                   dispersion=np.nan,
                                   voltage=np.nan,
                                   beta=np.nan,
                                   scan_type="BACKGROUND")
                self.background_image_signal.emit(raw_image)
                bgs.append(raw_image.T)

        mean_bg = np.mean(bgs, axis=0)
        return mean_bg

    def dispersion_scan(self, bg) -> dict[float, UFloat]:
        voltage = self.scan_request.dscan_tds_voltage
        slice_pos = self.scan_request.slice_pos
        self.set_tds_voltage(voltage)
        print("Doing dispersion scan at voltage", voltage)
        # print("Doing dispersion scan at amplitude", amplitude)
        widths = defaultdict(list)
        for setpoint in self.machine.scanner.scan.qscan.setpoints:
            self.set_quads(setpoint)
            time.sleep(self.scan_request.settings.quad_wait)
            sp_widths, _ = self.do_one_scan_setpoint(ScanType.DISPERSION,
                                                     dispersion=setpoint.dispersion,
                                                     voltage=voltage,
                                                     beta=setpoint.beta,
                                                     bg=bg,
                                                     slice_pos=slice_pos)
            widths[setpoint.dispersion].extend(sp_widths)
            # Get the average...
        widths = {dx: np.mean(widths) for dx, widths in widths.items()}
        return widths

    def tds_scan(self, bg) -> tuple[dict[float, UFloat], dict[float, UFloat]]:
        setpoint = self.machine.scanner.scan.tscan.setpoint
        slice_pos = self.scan_request.slice_pos
        self.set_quads(setpoint)
        time.sleep(3)
        widths = defaultdict(list)
        lengths = defaultdict(list)

        # We also calculate the bunch length at the maximum streak of the TDS Scan
        max(self.scan_request.voltages)

        for voltage in self.scan_request.voltages:
            self.set_tds_voltage(voltage)
            time.sleep(self.scan_request.settings.tds_amplitude_wait)
            sp_widths, sp_lengths = self.do_one_scan_setpoint(ScanType.TDS,
                                                              dispersion=setpoint.dispersion,
                                                              voltage=voltage,
                                                              beta=setpoint.beta,
                                                              bg=bg,
                                                              slice_pos=slice_pos)
            widths[voltage].extend(sp_widths)
            lengths[voltage].extend(sp_lengths)

        widths = {voltage: np.mean(widths) for voltage, widths in widths.items()}
        lengths = {voltage: np.mean(lengths) for voltage, lengths in lengths.items()}
        return widths, lengths

    def beta_scan(self, bg) -> dict[float, UFloat]:
        voltage = self.scan_request.dscan_tds_voltage
        slice_pos = self.scan_request.slice_pos
        self.set_tds_voltage(voltage)
        print("Doing beta scan at voltage", voltage)
        widths = defaultdict(list)

        for setpoint in self.machine.scanner.scan.bscan.setpoints:
            print(f"Starting beta scan setpoint={setpoint.beta}.")
            self.set_quads(setpoint)
            time.sleep(self.scan_request.settings.quad_wait)
            sp_widths, _ = self.do_one_scan_setpoint(ScanType.BETA,
                                                   setpoint.dispersion,
                                                   voltage=voltage,
                                                   beta=setpoint.beta,
                                                     bg=bg,
                                                     slice_pos=slice_pos)
            widths[setpoint.beta].extend(sp_widths)
            # Get the average...
        widths = {dx: np.mean(widths) for dx, widths in widths.items()}
        return widths

    def do_one_scan_setpoint(self, scan_type: ScanType, dispersion: float, voltage: float, beta: float, bg=0.0,
                             slice_pos=None):
        widths = [] # Result
        bunch_lengths = []
        # Output pandas dataframe of snapshots

        with self.snapshot_accumulator(scan_type, dispersion, voltage, beta) as accumulator:
            if self.kill:
                raise UserCancelledMeasurementException
            image_taker = self.take_screen_data(self.scan_request.images_per_setpoint, expect_beam=True)
            for raw_image in image_taker:
                processed_image = self.process_image(raw_image, scan_type,
                                                     dispersion=dispersion,
                                                     voltage=voltage,
                                                     beta=beta,
                                                     bg=bg,
                                                     slice_pos=slice_pos)
                accumulator.take_snapshot(raw_image,
                                          dispersion=dispersion,
                                          voltage=voltage,
                                          beta=beta,
                                          scan_type=str(scan_type))
                self.processed_image_signal.emit(processed_image)

                widths.append(processed_image.central_width)
                bunch_lengths.append(processed_image.sigma_z)

            return widths, bunch_lengths

    def set_quads(self, setpoint):
        self.machine.scanner.set_scan_setpoint_quads(setpoint)
        self.dispersion_sp_signal.emit(setpoint.dispersion)

    def set_tds_voltage(self, voltage):
        LOG.info(f"Setting TDS voltage: {voltage / 1e6} MV")
        # amplitude = self.scan_request.calibration.get_amplitude(voltage)
        self.machine.deflector.set_voltage(voltage)

    def take_screen_data(self, nbeam, expect_beam=True):
        self.scan_request.screen_name
        for _ in range(nbeam):
            is_beam_on = self.machine.is_beam_on()
            if expect_beam and not is_beam_on:
                raise MachineCancelledMeasurementException("Beam unexpectedly off.")
            elif not expect_beam and is_beam_on:
                raise MachineCancelledMeasurementException("Beam unexpectedly on.")

            time.sleep(0.2)
            raw_image = self.machine.screen.get_image_raw()
            yield raw_image

    def snapshot_accumulator(self, scan_type, dispersion, voltage, beta):
        shotter = self.machine.scanner.get_snapshotter()
        outdir = self.output_directory
        filename = make_snapshot_filename(scan_type=scan_type, dispersion=dispersion, voltage=voltage, beta=beta)
        return SnapshotAccumulator(shotter, outdir / filename)

    def background_data_accumulator(self):
        shotter = self.machine.scanner.get_snapshotter()
        outdir = self.output_directory
        filename = "background.pkl"
        return SnapshotAccumulator(shotter, outdir / filename)

    def process_image(self, image, scan_type: ScanType, dispersion: float, voltage: float, beta: float, bg=0, slice_pos=None) -> ProcessedImage:
        image = image.T # Flip to match control room..?  TODO
        image = filter_image(image, bg=bg, crop=True)

        _, means, sigmas = get_slice_properties(image)

        # sigma = get_central_slice_width_from_slice_properties(
        #     means, sigmas, padding=10, slice_pos=slice_pos
        # )

        central_width_row, sigma = get_selected_central_slice_width_from_slice_properties(means, sigmas, padding=10, slice_pos=slice_pos)

        # # Initially just pick middle slice
        # central_width_row = int(len(means) // 2)
        # print(central_width_row, len(means))
        # # elif slice_pos is None then we go with max energy slice
        # if slice_pos is None:
        #     central_width_row = np.argmin(means)
        # elif abs(slice_pos) > 0.5:
        #     raise ValueError("slice pos outside of [-0.5, 0.5].")
        # else:
        #     central_width_row = central_width_row + int(slice_pos * len(means) // 2)


        # print(slice_pos, central_width_row)
        
        r12_streaking = self.machine.optics.r12_streaking_from_tds_to_point(self.scan_request.screen_name)
        beam_energy = self.machine.optics.get_beam_energy() * 1e6 * e # MeV to Joules

        sigma_z = true_bunch_length_from_processed_image(image,
                                                         voltage=voltage,
                                                         r34=r12_streaking,
                                                         energy=beam_energy)

        return ProcessedImage(image, scan_type,
                              central_width=sigma,
                              central_width_row=central_width_row,
                              sigma_z=sigma_z,
                              dispersion=dispersion,
                              voltage=voltage,
                              beta=beta)


class MeasurementError(RuntimeError):
    pass


class InterruptedMeasurementException(MeasurementError):
    pass


class UserCancelledMeasurementException(InterruptedMeasurementException):
    pass


class MachineCancelledMeasurementException(InterruptedMeasurementException):
    pass


class MisconfiguredMeasurementException(MeasurementError):
    pass


def make_snapshot_filename(*, scan_type, dispersion, voltage, beta, **images):
    scan_string = scan_type.alt_name()
    voltage /= 1e6
    return f"{scan_string}-V={voltage=}MV_{dispersion=}m_{beta=}m.pkl"



class ScannerConfDialog(QtWidgets.QDialog):
    def __init__(self, defaults, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.defaults = defaults

        if not is_in_controlroom():
            self.defaults["output_directory_lineedit"] = "./esme-measurements"

        self.state = {}
        self.provisional_state = {}
        self.is_dirty = False
        self.restore_ui_initial_values()
        self.connect_buttons()


    def restore_ui_initial_values(self):
        uiconf = self.defaults
        quad_wait = uiconf["quad_sleep_spinbox"]
        tds_amplitude_wait = uiconf["tds_amplitude_wait_spinbox"]
        beam_on_wait = uiconf["beam_on_wait_spinbox"]
        self.ui.tds_amplitude_wait_spinbox.setValue(tds_amplitude_wait)
        self.ui.quad_sleep_spinbox.setValue(quad_wait)
        self.ui.beam_on_wait_spinbox.setValue(beam_on_wait)

        outdir = Path(uiconf["output_directory_lineedit"])
        outdir = str(Path(outdir.resolve()))

        self.ui.output_directory_lineedit.setText(outdir)
        self.state = deepcopy(self.defaults)
        self.provisional_state = deepcopy(self.defaults)

    def connect_buttons(self):
        sps = self._set_provisional_state

        self.ui.tds_amplitude_wait_spinbox.valueChanged.connect(lambda v: sps("tds_amplitude_wait_spinbox", v))
        self.ui.quad_sleep_spinbox.valueChanged.connect(lambda v: sps("quad_sleep_spinbox", v))
        self.ui.beam_on_wait_spinbox.valueChanged.connect(lambda v: sps("beam_on_wait_spinbox", v))
        self.ui.output_directory_lineedit.textEdited.connect(lambda v: sps("output_directory_lineedit", v))

        self.accepted.connect(self.okay_pressed)
        self.rejected.connect(self.cancel_pressed)

    def get_scan_settings(self):
        self.force_decision()

        s = self.state
        settings =  ScanSettings(quad_wait=s["quad_sleep_spinbox"],
                                 tds_amplitude_wait=s["tds_amplitude_wait_spinbox"],
                                 beam_on_wait=s["beam_on_wait_spinbox"],
                                 outdir=s["output_directory_lineedit"])
        LOG.debug("Getting settings: %s", settings)
        return settings

    def okay_pressed(self, close_window=False):
        LOG.debug("Settings Config OK pressed")
        self.is_dirty = False
        self.state = self.provisional_state
        if close_window:
            self.close()

    def cancel_pressed(self, close_window=False):
        LOG.debug("Settings Config Cancel pressed")
        self.is_dirty = False
        self.provisional_state = self.state
        if close_window:
            self.close()

    def force_decision(self):
        """If window is open then force the user to decide whether or
        not to accept the currently stored config values..."""
        if not self.isVisible():
            return
        if not self.is_dirty:
            self.close()
            return


        msg = QMessageBox()
        msg.setIcon(QMessageBox.Question)
        msg.setText("Unconfirmed Config Changes")
        msg.setInformativeText("Apply or discard changes to settings before starting measurement.")
        msg.setWindowTitle("Error")
        accept_changes_button = msg.addButton("Accept", QMessageBox.AcceptRole)
        reject_changes_button = msg.addButton("Discard", QMessageBox.RejectRole)

        accept_changes_button.clicked.connect(lambda: self.okay_pressed(close_window=True))
        reject_changes_button.clicked.connect(lambda: self.cancel_pressed(close_window=True))
        msg.exec_()



    def _set_provisional_state(self, widget_name, value):
        LOG.debug("Setting config box state %s = %s", widget_name, value)
        self.provisional_state[widget_name] = value
        self.is_dirty = True


@dataclass
class NewSetpointSignal:
    voltage: float
    dispersion: float
    beta: float

# def find_window(cls)
#     # Global function to find the (open) QMainWindow in application
#     app = QApplication.instance()
#     for widget in app.topLevelWidgets(): # or just instead .allWidgets()...?
#         if isinstance(widget, cls):
#             return widget
#     raise ValueError(f"Could not find widget of type: {cls}")




def display(pcl):
    import pickle
    with open(pcl, "rb") as f:
        import pickle
        p = pickle.load(f)

    app = QApplication(sys.argv)
    main_window = ScannerResultsDialog()

    main_window.post_measurement_result(p)

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


def main():
    # create the application
    app = QApplication(sys.argv)

    main_window = ScannerResultsDialog()
    # main_window = ScannerControl()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
