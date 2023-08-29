import sys

# Annoying hardcoded path on xfel machines that trumps any conda installation
try:
    sys.path.remove("/home/xfeloper/released_software/python/lib")
except ValueError:
    pass

import logging
import os
import pickle
import re
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyqtgraph as pg
import toml
from matplotlib import cm
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox

import esme.image as iana
from esme.analysis import FittedBeamParameters, OpticalConfig, transform_pixel_widths
from esme.calibration import TDSCalibrator, TDSVoltageCalibration
from esme.inout import (
    i1_dscan_config_from_scan_config_file,
    i1_tds_amplitudes_from_scan_config_file,
    make_data_taker,
    make_measurement_runner,
)
from esme.maths import ValueWithErrorT, linear_fit
from esme.measurement import (
    I1TDSCalibratingMachine,
    MeasurementPayload,
    ScanType,
    SetpointSnapshots,
)
from esme.plot import (
    TDS_AMPLITUDE_LABEL,
    TDS_CALIBRATION_LABEL,
    plot_calibrator_with_fits,
)

from .ui import esme_window, tds_calibration, tds_window

LOG = logging.getLogger(__name__)

# from .ui import


def is_in_controlroom():
    name = socket.gethostname()
    reg = re.compile("xfelbkr[0-9]\.desy\.de")
    return bool(reg.match(name))


# def get_outdir():
#     if is_in_controlroom():
#         outdir = Path("/Users/xfeloper/user/stwalker/esme-results/")
#         outdir.mkdir(parents=True, exist_ok=True)
#         return outdir
#     else:
#         return Path("./")


# def get_calibration_outdir():
#     if is_in_controlroom():
#         outdir = get_outdir() / "tds_calibration/"
#         outdir.mkdir(parents=True, exist_ok=True)
#     else:
#         Path("./"


def start_gui(scantoml, debug_mode, replay):
    """Main entry point to starting the GUI.  Reads from"""
    # the scan.toml
    # make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)
    path = os.path.join(
        os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png'
    )
    app.setWindowIcon(QtGui.QIcon(path))

    main_window = EnergySpreadMeasurementMainWindow(scantoml, replay_file=replay)

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


def start_tds_gui():
    # make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)
    path = os.path.join(
        os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png'
    )
    app.setWindowIcon(QtGui.QIcon(path))

    main_window = TDSMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


class FitWorker(QThread):
    def __init__(self, oconfig, reference_dispersion, reference_voltage, beam_energy):
        self.dscan_widths = defaultdict(list)
        # Now mapping voltages to widths here!
        self.tscan_widths = defaultdict(list)
        self.oconfig = oconfig
        self.reference_dispersion = reference_dispersion
        self.reference_voltage = reference_voltage
        self.beam_energy = beam_energy

    def receive_processed_image(self, result):
        if result.scan_type is ScanType.DISPERSION:
            self.dscan_widths[result.dispersion].append(cental_slice_width)

        elif result.scan_type is ScanType.TDS:
            self.tscan_widths[result.voltage].append(cental_slice_width)

    def generate_fitted_parameters(self):
        dispersions = np.array(self.dscan_widths.keys())
        dwidths = self.dscan_widths.values()

        voltage = np.array(self.tscan_widths.keys())
        twidths = self.tscan_widths.values()

        tdata = TDSScanData(voltage, twidths)
        ddata = DispersionScanData(dispersions, dwidths)

        sem = SliceEnergySpreadMeasurement(
            dscan,
            tscan,
            self.oconfig,
            self.reference_dispersion,
            self.reference_voltage,
            self.beam_energy,
        )

        return sem.all_fit_parameters()


class BackgroundData:
    def __init__(self, nbg):
        # Expected number of background images..  not sure this is really necessary but..
        self.nbg = nbg
        self.images = []
        self._mean_bg_im = None

    def append(self, image):
        self.images.append(image)

    def mean_bg_im(self):
        if self._mean_bg_im is not None and len(self.images) == self.nbg:
            return self._mean_bg_im
        self._mean_bg_im = np.mean(self.images, axis=0)
        return self._mean_bg_im


@dataclass
class ImageProcessingResult:
    central_slice_width: float
    amplitude: float
    voltage: float
    dispersion: float
    scan_type: ScanType


class ImageAnalysisThread(QThread):
    processed_image_result_signal = pyqtSignal(ImageProcessingResult)
    processed_image_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)

    def __init__(self, tds_percentages, dispersions, nbg, nbeam, tds_calibration=None):
        super().__init__()
        self.nbg = nbg
        self.nbeam = nbeam

        self.ntotal = (nbg + nbeam) * (len(tds_percentages) + len(dispersions))
        self.n_processed = 0
        self.tds_calibration = tds_calibration

        self.dscan_background = defaultdict(lambda: BackgroundData(nbg))
        self.tscan_background = defaultdict(lambda: BackgroundData(nbg))

        self.dscan_dfs = defaultdict(list)
        self.tscan_dfs = defaultdict(list)

    def append_background_image(self, payload: MeasurementPayload):
        assert payload.is_bg, "Not a background image payload!"
        if payload.scan_type is ScanType.DISPERSION:
            self.dscan_background[payload.dispersion_setpoint].append(
                payload.image.image
            )
        elif payload.scan_type is ScanType.TDS:
            self.tscan_background[payload.tds_percentage].append(payload.image.image)
        else:
            raise RuntimeError("Malformed background payload")
        self.increment_progress()

    def process_beam_image(self, payload: MeasurementPayload):
        assert not payload.is_bg, "Not a beam image!"
        bg_im = self.get_mean_background_for_beam_image(payload)

        processed_image = iana.process_image(payload.image.image, bg_im)
        x, means, sigmas = iana.get_slice_properties(processed_image)
        centre_index = means.argmin()

        padding = 10

        sigma = np.mean(sigmas[centre_index - padding : centre_index + padding])

        voltage = None
        if self.tds_calibration:
            voltage = self.tds_calibration.get_voltage(payload.tds_percentage)

        result = ImageProcessingResult(
            central_slice_width=sigma,
            amplitude=payload.tds_percentage,
            voltage=voltage,
            dispersion=payload.dispersion_setpoint,
            scan_type=payload.scan_Type,
        )

        self.processed_image_result_signal.emit(result)
        self.increment_progress()

    def process_payload(self, payload):
        if payload.is_bg:
            self.append_background_image(payload)
        elif not payload.is_bg:
            self.process_beam_image(payload)

    def get_mean_background_for_beam_image(self, payload):
        if payload.scan_type is ScanType.DISPERSION:
            return self.dscan_background[payload.dispersion_setpoint].mean_bg_im()
        elif payload.scan_type is ScanType.TDS:
            return self.tscan_background[payload.tds_percentage].mean_bg_im()
        else:
            raise RuntimeError("Malformed background payload")

    def increment_progress(self):
        self.n_processed += 1
        self.progress_signal.emit(self.n_processed)


class TDSScanData:
    def __init__(self, voltages, slice_widths):
        self.voltages = voltages
        self.slice_widths = slice_widths

    def get_mean_slice_widths_and_errors(self):
        widths_with_errors = [np.mean(x) for x in self.slice_widths]
        widths = widths_with_errors[..., 0]
        errors = widths_with_errors[..., 1]
        return widths, errors

    def fit(self):
        widths, errors = self.get_mean_slice_widths_and_errors()
        voltages2 = self.voltages**2
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v


class DispersionScanData:
    def __init__(self, dispersions, slice_widths):
        self.dispersions = dispersions
        self.slice_widths = slice_widths

    def get_mean_slice_widths_and_errors(self):
        widths_with_errors = [np.mean(x) for x in self.slice_widths]
        widths = widths_with_errors[..., 0]
        errors = widths_with_errors[..., 1]
        return widths, errors

    def fit(self):
        widths, errors = self.get_mean_slice_widths_and_errors()
        dx2 = self.dispersions**2
        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v


class SliceEnergySpreadMeasurement:
    def __init__(
        self,
        dscan: DispersionScanData,
        tscan: TDSScanData,
        optical_config: OpticalConfig,
        reference_dispersion,
        reference_voltage,
        beam_energy,
    ):
        self.dscan = dscan
        self.tscan = tscan
        self.oconfig = optical_config
        self.reference_voltage = reference_voltage
        self.reference_dispersion = reference_dispersion
        self.beam_energy = beam_energy

    def dispersion_scan_fit(self) -> ValueWithErrorT:
        widths, errors = self.dscan.max_energy_slice_widths_and_errors(padding=10)
        dx2 = self.dscan.dx**2
        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v

    def tds_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths, errors = self.tscan.max_energy_slice_widths_and_errors(padding=10)
        voltages2 = self.tscan.voltage**2
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v

    def all_fit_parameters(self) -> FittedBeamParameters:
        a_v, b_v = self.tscan.fit()
        a_d, b_d = self.dscan.fit()

        # Values and errors, here we just say there is 0 error in the
        # dispersion and voltage, not strictly true.
        energy = self.beam_energy, 0.0  # in eV
        # dispersion = self.oconfig.reference_dispersion, 0.
        # voltage = self.oconfig.reference_voltage, 0.

        a_beta = b_beta = None

        return FittedBeamParameters(
            a_v=a_v,
            b_v=b_v,
            a_d=a_d,
            b_d=b_d,
            reference_energy=energy,
            reference_dispersion=self.reference_dispersion,
            reference_voltage=self.reference_voltage,
            oconfig=self.oconfig,
            a_beta=a_beta,
            b_beta=b_beta,
        )

        # if payload.is_bg and payload.scan_type is ScanType.DISPERSION:


# TODO: SOMEWHERE WRITE IMAGES TO FILE!

# def run(self):
#     while not self.kill:


class EnergySpreadMeasurementMainWindow(QMainWindow):
    def __init__(self, scanfile, replay_file):
        super().__init__()
        self.ui = esme_window.Ui_MainWindow()
        self.ui.setupUi(self)
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)

        LOG.info("Starting Energy Spread Measurement GUI...")

        # Just have path, that way we can update on the fly.
        # Maybe slower at runtime but better for active development
        self.scanfile = scanfile

        self.ui.start_measurement_button.clicked.connect(self.start_measurement)
        self.ui.tds_calibration_button.clicked.connect(self.show_calibration_window)

        self.replay_file = replay_file

        self.timer = QTimer()
        # Function that does nothing so that at least python code is
        # processed (namely at least interrupts).
        self.timer.timeout.connect(lambda: None)
        # self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)

        self.sw = self.ui.screen_widget

        self.sw.roi.hide()
        self.sw.ui.menuBtn.hide()
        self.sw.ui.roiBtn.hide()
        self.sw.ui.roiPlot.hide()

        cmap = pg.colormap.get('viridis')
        self.sw.setColorMap(cmap)

        self.tdsw = None

        self.tds_calibration = None

        self.sw.show()

    def closeEvent(self, event):
        # If this window is closed then also close the daughter TDS
        # calibration window if it exists.
        if self.tdsw:
            self.tdsw.close()

    def should_save_data(self):
        return bool(self.ui.save_data_checkbox.checkState())

    def get_outdir(self):
        text = self.ui.lineEdit.text()
        return get_outdir() / Path(text)

    def start_measurement(self):
        nbeam = self.nbeam_images()
        nbg = self.nbg_images()

        print(nbeam, nbg)

        location = self.measurement_location()

        self.data_taking_thread = DataTakingThread(
            scanfile=self.scanfile,
            location=location,
            outdir=get_outdir(),
            replay_file=self.replay_file,
            nbg=nbg,
            nbeam=nbeam,
            save=self.should_save_data(),
        )
        # self.data_taking_thread.payload_signal.connect(self.process_payload)

        dispersions = i1_dscan_config_from_scan_config_file(self.scanfile).dispersions
        percentages = i1_tds_amplitudes_from_scan_config_file(self.scanfile)
        self.ana_thread = ImageAnalysisThread(
            percentages, dispersions, nbg, nbeam, tds_calibration=self.tds_calibration
        )
        self.ana_thread.progress_signal.connect(
            self.ui.measurement_progress_bar.setValue
        )
        self.data_taking_thread.payload_signal.connect(self.ana_thread.process_payload)
        self.ana_thread.processed_image_signal.connect(self.update_image_from_payload)

        # Without a TDS calibration then the fitting functionality is
        # useless / makes no sense.  so just return.

        if self.tds_calibration is not None:
            oconf = optics_config_from_toml("i1", self.scanfile)

            conf = toml.load(self.scanfile)

            beam_energy = conf["i1"]["tds"]["optics"]["beam_energy"]

            reference_voltage = self.calibration.get_voltage(
                conf["i1"]["tds"]["reference_amplitude"]
            )
            reference_dispersion = self.calibration.get_voltage(
                conf["i1"]["tds"]["scan_dispersion"]
            )

            self.fit_worker_thread = FitWorker(
                oconf,
                reference_dispersion=reference_dispersion,
                reference_voltage=reference_voltage,
                beam_energy=beam_energy,
            )

            self.ana_thread.processed_image_result_signal.connect(
                self.fit_worker_thread.receive_processed_image
            )

        self.ana_thread.start()
        self.data_taking_thread.start()

        if self.tds_calibration is not None:
            self.fit_worker_thread.start()

    def print_result(self):
        pass

    def update_image_from_payload(self, payload):
        # image_event = payload.image.astype("float64")
        image_event = payload.image.image
        p = payload
        ampl = payload.tds_percentage
        disp = payload.dispersion_setpoint
        LOG.info("Posting image: bg:{p.is_bg}, tds:{ampl}, {dispersion_setpoint}")
        self.sw.setImage(image_event)

    def get_image(self):
        pass

    def nbeam_images(self):
        return self.ui.beam_shots_spinner.value()

    def nbg_images(self):
        return self.ui.bg_shots_spinner.value()

    def measurement_location(self):
        if self.ui.i1_radio_button.isChecked():
            return "i1"
        elif self.ui.b2_radio_button.isChecked():
            return "b2"

    def show_calibration_window(self):
        self.tdsw = TDSCalibrationWindow(parent=self)
        self.tdsw.show()

    @property
    def tds_calibration(self):
        return self._tds_calibration

    @tds_calibration.setter
    def tds_calibration(self, calibration):
        LOG.info(f"Setting TDS Voltage Calibration: {calibration}")
        if calibration is None:
            self._tds_calibration = None
            return
        LOG.info(f"TDS Amplitudes: {calibration.amplitudes}")
        vstring = np.array2string(
            calibration.voltages,
            separator=", ",
            max_line_width=400,
            floatmode="fixed",
            precision=0,
        )
        LOG.info(f"Derived Voltages: {vstring}")
        self._tds_calibration = calibration


class DataTakingThread(QThread):
    payload_signal = pyqtSignal(MeasurementPayload)
    finished_signal = pyqtSignal()

    def __init__(self, *, scanfile, location, outdir, replay_file, nbg, nbeam, save):
        super().__init__()
        self.nbg = nbg
        self.nbeam = nbeam
        self.runner = make_data_taker(
            fconfig=scanfile,
            machine_area=location,
            outdir=get_outdir(),
            measure_dispersion=False,
            replay_file=replay_file,
        )
        self.outdir = Path(outdir)
        self.save = save

        self.dscan_dfs = defaultdict(pd.DataFrame)
        self.dscan_tds_amplitude = None
        self.tscan_dispersion = None
        self.tscan_dfs = defaultdict(pd.DataFrame)

    def run(self):
        said_save_already = False

        for i, payload in enumerate(
            self.runner.measure(bg_shots=self.nbg, beam_shots=self.nbeam)
        ):
            self.payload_signal.emit(payload)

            if payload.scan_type is ScanType.DISPERSION:
                self.dscan_tds_amplitude = payload.tds_percentage

            if payload.scan_type is ScanType.TDS:
                self.tscan_dispersion = payload.dispersion_setpoint

            if self.save:
                image_filename = get_image_filename_from_payload(self.outdir, payload)
                pcl_fname = Path(str(image_filename) + ".pcl")
                with pcl_fname.open("wb") as f:
                    print(image_filename)
                    pickle.dump(payload.image.image, f)
                png_fname = pcl_fname.with_suffix(".png")
                matplotlib.image.imsave(png_fname, payload.image.image)
                if not said_save_already:
                    LOG.info(f"Written images to {pcl_fname.parent}")
                    said_save_already = True

                self.dscan_dfs[payload.dispersion_setpoint]

            self.append_snapshot(payload)

        if self.save:
            self.save_snapshots()

        self.finished_signal.emit()

    def save_snapshots(self):
        for dispersion, snapshots in self.dscan_dfs.items():
            sp = SetpointSnapshots(
                snapshots,
                scan_type=ScanType.DISPERSION,
                dispersion_setpoint=dispersion,
                measured_dispersion=(dispersion, 0),
            )
            fname = self.make_snapshots_filename(sp)

            with fname.open("wb") as f:
                pickle.dump(sp, f)
                print("written", sp, "@", f)

        for tds_amplitude, snapshots in self.tscan_dfs.items():
            sp = SetpointSnapshots(
                snapshots,
                scan_type=ScanType.TDS,
                dispersion_setpoint=self.tscan_dispersion,
                measured_dispersion=(self.tscan_dispersion, 0),
            )
            fname = self.make_snapshots_filename(sp)

            with fname.open("wb") as f:
                pickle.dump(sp, f)
                print("written", sp, "@", f)

    def make_snapshots_filename(self, snapshot):
        """Make a human readnable name for the output dataframe"""
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        ampl = self.dscan_tds_amplitude
        dispersion = snapshot.dispersion_setpoint
        scan_type = snapshot.scan_type
        fname = f"{timestamp}>>{scan_type}>>D={dispersion},TDS={ampl}%.pcl"
        return self.outdir / fname

    def append_snapshot(self, payload):
        if payload.scan_type is ScanType.DISPERSION:
            df = self.dscan_dfs[payload.dispersion_setpoint]
            self.dscan_dfs[payload.dispersion_setpoint] = pd.concat(
                [df, payload.snapshot]
            )

        elif payload.scan_type is ScanType.TDS:
            df = self.tscan_dfs[payload.tds_percentage]
            self.tscan_dfs[payload.tds_percentage] = pd.concat([df, payload.snapshot])

        else:
            raise RuntimeError("Malformed snapshot in datatakingthread.append_snapshot")


def get_image_filename_from_payload(outdir, payload):
    channel = payload.image.channel
    dirname = f"images-{payload.image.screen_name()}"
    fstem = payload.image.name()

    return outdir / dirname / fstem


from esme.analysis import transform_pixel_widths
from esme.calibration import TDS_FREQUENCY

TDS_PERIOD = 1 / TDS_FREQUENCY


@dataclass
class TDSCalibrationPayload:
    tds_amplitude: float
    raw_phase: list
    raw_com: list

    # slope1: tuple[list, list]
    # slope2: tuple[list, list]
    def smoothed_signal(self):
        return smooth(self.raw_phase, self.raw_com, window=10)

    def section1(self):
        x, y = self.smoothed_signal()
        first, _ = get_truncated_longest_sections(x, y, com_window_size=20)
        return first

    def section2(self):
        x, y = self.smoothed_signal()
        _, second = get_truncated_longest_sections(x, y, com_window_size=20)
        return second

    def gradient1(self):
        phase, com = self.section1()
        com, _ = transform_pixel_widths(
            com, np.zeros_like(com), pixel_units="m", dimension="y", to_variances=False
        )
        grad = np.gradient(com, phase)
        seconds_per_degree = TDS_PERIOD / 360
        grad2 = grad / seconds_per_degree
        return grad2

    def gradient2(self):
        phase, com = self.section2()
        com, _ = transform_pixel_widths(
            com, np.zeros_like(com), pixel_units="m", dimension="y", to_variances=False
        )
        grad = np.gradient(com, phase)  # in m / degree
        seconds_per_degree = TDS_PERIOD / 360
        grad2 = grad / seconds_per_degree
        return grad2


class TDSTimeCalibrationWorker(QThread):
    calibration_payload_signal = pyqtSignal(TDSCalibrationPayload)
    calibration_slope_signal = pyqtSignal(float)

    def __init__(self, machine, amplitudes):
        super().__init__()
        self.machine = machine
        self.amplitudes = amplitudes

    def run(self):
        for tds_amplitude in self.amplitudes:
            self.machine.tds.set_amplitude(tds_amplitude)
            time.sleep(0.5)
            phases = []
            ycoms = []
            for phase, ycom in self.machine.get_coms():
                phases.append(phase)
                ycoms.append(ycom)
            payload = TDSCalibrationPayload(tds_amplitude, phases, ycoms)
            self.calibration_payload_signal.emit(payload)


from .calibrate_tds import (
    get_longest_two_monotonic_intervals,
    get_truncated_longest_sections,
    smooth,
)


class TDSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = tds_window.Ui_MainWindow()
        self.ui.setupUi(self)
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.log_output_widget.append)

        self.machine = I1TDSCalibratingMachine()

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_tds_from_machine)
        self.timer2hz = QTimer()
        self.timer2hz.timeout.connect(self.show_beam)
        self.timer.start(100)
        self.timer2hz.start(500)

        # embed()
        self.ui.phase_spin_box_2.valueChanged.connect(self.machine.tds.set_phase)
        self.ui.amplitude_spin_box.valueChanged.connect(self.machine.tds.set_amplitude)
        self.ui.on_beam_push_button.clicked.connect(self.machine.tds.switch_on_beam)
        self.ui.off_beam_push_button.clicked.connect(self.machine.tds.switch_off_beam)
        self.ui.plus_180_phase.clicked.connect(
            lambda: self.machine.tds.set_phase(180 + self.machine.tds.read_sp_phase())
        )
        self.ui.minus_180_phase.clicked.connect(
            lambda: self.machine.tds.set_phase(-180 + self.machine.tds.read_sp_phase())
        )
        self.ui.start_voltage_calibration_button.clicked.connect(
            self.do_time_calibration
        )
        self.add_plot()
        # embed()

    def read_tds_from_machine(self):
        self.ui.phase_spin_box_2.setValue(self.machine.tds.read_sp_phase())
        self.ui.amplitude_spin_box.setValue(self.machine.tds.read_sp_amplitude())
        on_beam = self.machine.tds.is_on_beam()
        self.ui.on_beam_push_button.setChecked(on_beam)
        self.ui.off_beam_push_button.setChecked(not on_beam)

    def show_beam(self):
        image = self.machine.get_screen_image().astype("float64")
        self.img.setImage(image)

    def get_amplitudes(self):
        line = self.ui.amplitudes_line_edit.text()
        if not line:
            return []
        line = line.split(",")
        return [float(x) for x in line]

    def do_time_calibration(self):
        self.time_calibration_thread = TDSTimeCalibrationWorker(
            self.machine, self.get_amplitudes()
        )
        self.time_calibration_thread.calibration_payload_signal.connect(
            self.update_time_calibration_plot
        )
        self.time_calibration_thread.start()

    def update_time_calibration_plot(self, payload):
        p, y = payload.raw_phase, payload.raw_com
        p1, y1 = payload.section1()
        p2, y2 = payload.section2()
        (l1,) = self.ui.voltage_calibration_plot.axes.plot(
            p, y, label=f"{payload.tds_amplitude}%, raw", linestyle="--", alpha=0.5
        )
        self.ui.voltage_calibration_plot.axes.plot(p1, y1, color=l1.get_color())
        self.ui.voltage_calibration_plot.axes.plot(p2, y2, color=l1.get_color())

        self.ui.voltage_calibration_plot.draw()
        self.ui.voltage_calibration_plot.axes.legend()

        print(payload.gradient1(), payload.gradient2())

    def add_plot(self):
        # win = pg.GraphicsLayoutWidget()
        # layout = QtWidgets.QGridLayout()
        # self.ui.widget.setLayout(layout)
        # layout.addWidget(win)

        win = self.ui.screen_display_widget

        self.img_plot = win.addPlot()
        self.img_plot.clear()

        # self.img_plot.setLabel('left', "N bunch", units='')
        # self.img_plot.setLabel('bottom', "", units='eV')
        self.img_plot.hideAxis('left')
        self.img_plot.hideAxis('bottom')
        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis')  # "nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(
            np.ndarray
        )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt

        # Custom ROI for selecting an image region
        # self.roi = pg.ROI([20, 20], [6, 50])
        # self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        # self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        # self.img_plot.addItem(self.roi)
        # self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        # iso = pg.IsocurveItem(level=0.8, pen='g')
        # iso.setParentItem(self.img)
        # iso.setZValue(5)

        # self.p1 = win.addPlot(colspan=1)
        # self.p1.setLabel('right', "E [eV]", units='')
        # self.p1.hideAxis('left')
        # self.p1.getViewBox().invertX(True)
        # self.p1.setMaximumWidth(200)
        # self.p1.setYLink(self.img_plot)
        # #self.p1.setMaximumHeight(50)

        # # Contrast/color control
        # #hist = pg.HistogramLUTItem()
        # #hist.setImageItem(self.img)
        # #win.addItem(hist)

        # # Draggable line for setting isocurve level
        # #isoLine = pg.InfiniteLine(angle=0, movable=True, pen='g')
        # #hist.vb.addItem(isoLine)
        # #hist.vb.setMouseEnabled(y=False)  # makes user interaction a little easier
        # #isoLine.setValue(0.8)
        # #isoLine.setZValue(1000)  # bring iso line above contrast controls

        # # Another plot area for displaying ROI data
        # win.nextRow()
        # self.p2 = win.addPlot(colspan=1)
        # self.p2.setLabel('bottom', "t [ps]", units='')
        # self.p2.setMaximumHeight(200)
        # self.p2.setXLink(self.img_plot)
        # win.resize(800, 800)
        # win.show()

        # # Generate image data
        # self.image_event = np.random.normal(size=(200, 100))
        # self.image_event[20:80, 20:80] += 2.
        # self.image_event = pg.gaussianFilter(self.image_event, (3, 3))
        # self.image_event += np.random.normal(size=(200, 100)) * 0.1
        # self.img.setImage(self.image_event)
        # # set position and scale of image
        # #self.img.scale(0.2, 0.2)
        # #self.img.translate(-50, 0)

        # hist.setLevels(data.min(), data.max())

        # build isocurves from smoothed data
        # iso.setData(pg.gaussianFilter(data, (2, 2)))

        # set position and scale of image
        # self.img.scale(0.2, 0.2)
        # self.img.translate(-50, 0)

        # zoom to fit imageo
        # self.img_plot.autoRange()
        # Apply the colormap
        self.img.setLookupTable(lut)


# class EnergySpreadMeasurementReplayMainWindow(QMainWindow)


class TDSCalibrationWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent = parent
        self.ui = tds_calibration.Ui_TDSCalibrationWindow()
        self.ui.setupUi(self)

        self.try_and_plot_calibration()

        self.mi = I1TDSCalibratingMachine("./")

        # self.timer2hz = QTimer()
        # self.timer2hz.timeout.connect(self.show_beam)
        # self.timer.start(100)
        # self.timer2hz.start(500)

        self.active_snapshot = None
        self.ui.update_voltage_button.clicked.connect(self.display_voltage)

        self.ui.apply_calibration_button.clicked.connect(
            self.apply_calibration_to_parent_widget
        )
        self.ui.save_tds_calibration_button.clicked.connect(self.save_calibration_file)
        self.ui.load_tds_calibration_button.clicked.connect(self.open_calibration_file)
        self.ui.apply_calibration_button.clicked.connect(
            self.send_calibrator_to_parent_widget
        )

        if self.parent is None:
            self.ui.apply_calibration_button.disabled()

    def try_and_plot_calibration(self):
        try:
            calib = self.text_box_to_calibrator()
        except (KeyError, toml.TomlDecodeError):
            return

        ax = self.ui.widget_plot_1.axes
        ax.cla()
        plot_calibrator_with_fits(calib, fig=self.ui.widget_plot_1.figure, ax=ax)
        self.ui.widget_plot_1.figure.tight_layout()
        self.ui.widget_plot_1.draw()

    def update_snapshot(self):
        if is_in_controlroom():
            snapshot, _ = self.mi.get_machine_snapshot()
            self.active_snapshot = snapshot
            print(self.active_snapshot, "WOW?")
        else:
            self.active_snapshot = pd.read_pickle(
                "/Users/stuartwalker/repos/emem/pyBigBro_28_11_2022/dx_1.2m_dispersion_one_snapshot.pcl"
            )

    def text_box_to_conf_dict(self):
        text = self.ui.calibration_info.toPlainText()
        conf = toml.loads(text)
        return conf

    def text_box_to_calibrator(self):
        conf = self.text_box_to_conf_dict()
        ampls = conf["percentages"]
        tds_slopes = conf["tds_slopes"]
        units = conf["tds_slope_units"]

        calib = TDSCalibrator(
            ampls, tds_slopes, dispersion_setpoint=1.2, tds_slope_units=units
        )

        return calib

    def text_box_to_voltage_calibration(self):
        calibrator = self.text_box_to_calibrator()

        if self.active_snapshot is None:
            self.offer_to_update_machine_snapshot()
        if self.active_snapshot is None:
            return

        voltages = calibrator.get_voltage(calibrator.percentages, self.active_snapshot)
        voltage_calibration = TDSVoltageCalibration(calibrator.percentages, voltages)

        return voltage_calibration

    def offer_to_update_machine_snapshot(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("New Machine Snapshot")
        msg.setText(
            "Would you like to take a new machine snapshot for calculating R<sub>34</sub> (R<sub>12</sub>)?"
        )
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)

        button = msg.exec()
        if button == QMessageBox.Yes:
            self.update_snapshot()

    def display_voltage(self):
        calib = self.text_box_to_calibrator()

        self.offer_to_update_machine_snapshot()

        if self.active_snapshot is None:
            return

        voltage = calib.get_voltage(calib.percentages, self.active_snapshot)

        ax = self.ui.widget_plot_2.axes
        self.try_and_plot_calibration()
        ax.cla()
        ax.plot(calib.percentages, abs(voltage * 1e-6))
        ax.set_xlabel(r"TDS Amplitude / %")
        ax.set_ylabel(r"$|V_\mathrm{TDS}|$ / MV")
        self.ui.widget_plot_2.figure.tight_layout()
        self.ui.widget_plot_2.draw()
        vstring = np.array2string(
            abs(voltage * 1e-6),
            separator=", ",
            max_line_width=200,
            floatmode="fixed",
            precision=4,
        )
        conf_dict = self.text_box_to_conf_dict()
        conf_dict["tds_voltage"] = vstring
        conf_dict["tds_voltage_units"] = "MV"
        self.post_config_to_text_box(conf_dict)

    def post_config_to_text_box(self, confdict):
        text = toml.dumps(confdict)
        self.ui.calibration_info.clear()
        self.ui.calibration_info.append(text)

    def apply_calibration_to_parent_widget(self):
        if self.parent is None:
            return
        parent_name = self.parent.windowTitle()
        # self.tds_calibrator = self.

    def save_calibration_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # from IPython import embed; embed()
        # default_dir = get_directory
        # fileName, _ = QFileDialog.getOpenFileName(self, "Save TDS Calibration", directory=get_calibration_outdir(),filter="toml files (*.toml)", initialFilter="toml files", options=options)
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save TDS Calibration",
            directory=get_calibration_outdir(),
            filter="toml files (*.toml)",
            initialFilter="toml files",
            options=options,
        )
        fname = Path(fname)
        if fname:
            if fname.is_dir():
                return

            if fname.suffix != ".toml":
                fname = fname.with_suffix(".toml")

            with open(fname, "w") as f:
                f.write(self.ui.calibration_info.toPlainText())

    def open_calibration_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # from IPython import embed; embed()
        # default_dir = get_directory
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Load TDS Calibration",
            directory=get_calibration_outdir(),
            filter="toml files (*.toml)",
            initialFilter="toml files",
            options=options,
        )
        # fname, _ = QFileDialog.getSaveFileName(self, "Save TDS Calibration", directory=get_calibration_outdir(),filter="toml files (*.toml)", initialFilter="toml files", options=options)
        if fname:
            with open(fname, "r") as f:
                new_text = f.read()
                self.ui.calibration_info.clear()
                self.ui.calibration_info.insertPlainText(new_text)

    def send_calibrator_to_parent_widget(self):
        calibration = self.text_box_to_voltage_calibration()
        parent = self.parent
        parent.tds_calibration = calibration
