from pathlib import Path
import os
import sys
from matplotlib import cm
import numpy as np
import socket
import re
from esme.inout import i1_dscan_config_from_scan_config_file, i1_tds_amplitudes_from_scan_config_file
import time

from PyQt5.QtWidgets import QFrame, QMainWindow, QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer

from esme.measurement import MeasurementPayload, ScanType, I1TDSCalibratingMachine
import esme.image as iana

import pyqtgraph as pg


from esme.inout import make_measurement_runner, make_data_taker
from .ui import esme_window, tds_window

import logging

LOG = logging.getLogger(__name__)

# from .ui import


def is_in_controlroom():
    name = socket.gethostname()
    reg = re.compile("xfelbkr[0-9]\.desy\.de")
    return bool(reg.match(name))

def get_outdir():
    if is_in_controlroom:
        return Path("/Users/xfeloper/user/stwalker/esme-results/")


def start_gui(scantoml, debug_mode, replay):
    # make pyqt threadsafe
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    # create the application
    app = QApplication(sys.argv)
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png')
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
    path = os.path.join(os.path.dirname(sys.modules[__name__].__file__), 'gui/hirex.png')
    app.setWindowIcon(QtGui.QIcon(path))

    main_window = TDSMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


def get_outdir():
    return Path("./")


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)
    def __init__(self):
        super().__init__()

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)

from collections import defaultdict


class EnergySpreadAnalysisWorker(QThread):
    finished = pyqtSignal()
    progress_signal = pyqtSignal(int)
    processed_image_signal = pyqtSignal(np.ndarray)

    def __init__(self, tds_percentages, dispersions, nbg, nbeam):
        super().__init__()
        self.nbg = nbg
        self.nbeam = nbeam

        self.ntotal = (nbg + nbeam) * (len(tds_percentages) + len(dispersions))
        self.n_processed = 0

        # self.tds_percentages = percentage
        # self.dispersions = dispersions

        self.dscan_dfs = defaultdict(list)
        self.tscan_dfs = defaultdict(list)

        self.dscan_bg_images = defaultdict(list)
        self.tscan_bg_images = defaultdict(list)

        self.dscan_widths = defaultdict(list)
        self.tscan_widths = defaultdict(list)

    def process_payload(self, payload):
        st = payload.scan_type
        df = payload.snapshot
        im = payload.image
        is_bg = payload.is_bg


        if st is ScanType.DISPERSION:
            self.dscan_dfs[payload.dispersion_setpoint].append(df)

            if is_bg:
                self.dscan_bg_images[payload.dispersion_setpoint].append(im)
                self.processed_image_signal.emit(im)
            else:
                bg_images = self.dscan_bg_images[payload.dispersion_setpoint]
                self.processed_image_signal.emit(im)
                # self.processed_image_signal.emit(crop_and_clean_image(im, bg_images))
                self.dscan_widths[payload.dispersion_setpoint].append(get_central_slice(im, bg_images))

        elif st is ScanType.TDS:
            self.tscan_dfs[payload.tds_percentage].append(df)

            if is_bg:
                self.tscan_bg_images[payload.tds_percentage].append(im)
                self.processed_image_signal.emit(im)
            else:
                bg_images = self.tscan_bg_images[payload.tds_percentage]
                self.processed_image_signal.emit(im)
                # self.processed_image_signal.emit(crop_and_clean_image(im, bg_images))
                self.tscan_widths[payload.tds_percentage].append(get_central_slice(im, bg_images))

        self.n_processed += 1
        print(int(100 * self.n_processed / self.ntotal))
        self.progress_signal.emit(int(100 * self.n_processed / self.ntotal))

        # if payload.is_bg and payload.scan_type is ScanType.DISPERSION:

# TODO: SOMEWHERE WRITE IMAGES TO FILE!

    # def run(self):
    #     while not self.kill:

def crop_and_clean_image(beam_image, background_images):
    mean_bg_im = np.mean(background_images, axis=0)
    return iana.process_image(beam_image, mean_bg_im)

def get_central_slice(beam_image, background_images):
    return None
    mean_bg_im = np.mean(bgs, axis=0)
    x, means, sigmas = iana.get_slice_properties(image)
    centre_index = means.argmin()
    sigma = np.mean(sigmas[centre_index - padding : centre_index + padding])
    # should have an error here too...
    return sigma

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
        self.replay_file = replay_file

        self.timer = QTimer()
        # Function that does nothing so that at least python code is
        # processed (namely at least interrupts).
        self.timer.timeout.connect(lambda: None)
        # self.timer.timeout.connect(self.update_plot)
        self.timer.start(100)


        self.sw = self.ui.screen_widget
        # from IPython import embed; embed()

        self.sw.roi.hide()
        self.sw.ui.menuBtn.hide()
        self.sw.ui.roiBtn.hide()
        self.sw.ui.roiPlot.hide()

        cmap = pg.colormap.get('viridis')
        self.sw.setColorMap(cmap)



        self.sw.show()

    def start_measurement(self):
        nbeam = self.nbeam_images()
        nbg = self.nbg_images()
        location = self.measurement_location()

        self.thread = DataTakingThread(self.scanfile, location, get_outdir(), self.replay_file, nbg, nbeam)
        # self.thread.payload_signal.connect(self.process_payload)

        dispersions = i1_dscan_config_from_scan_config_file(self.scanfile).dispersions
        percentages = i1_tds_amplitudes_from_scan_config_file(self.scanfile)
        self.ana_thread = EnergySpreadAnalysisWorker(percentages, dispersions, nbg, nbeam)
        self.ana_thread.progress_signal.connect(self.ui.measurement_progress_bar.setValue)
        self.thread.payload_signal.connect(self.ana_thread.process_payload)

        self.ana_thread.processed_image_signal.connect(self.process_payload)

        self.ana_thread.start()
        self.thread.start()

    def process_payload(self, image):
        # image_event = payload.image.astype("float64")
        image_event = image
        self.sw.setImage(image_event)

    def get_image(self):
        pass

    def nbeam_images(self):
        return self.ui.beam_shots_spinner.value()

    def nbg_images(self):
        return self.ui.bg_shots_spinner.value()

    def measurement_location(self):
        if self.ui.i1d_radio_button.isChecked():
            return "i1"
        elif self.ui.b2d_radio_button.isChecked():
            return "b2"

class DataTakingThread(QThread):
    payload_signal = pyqtSignal(MeasurementPayload)
    def __init__(self, scanfile, location, outdir, replay_file, nbg, nbeam):
        super().__init__()
        self.nbg = nbg
        self.nbeam = nbeam
        self.runner = make_data_taker(fconfig=scanfile,
                                      machine_area=location,
                                      outdir=get_outdir(),
                                      measure_dispersion=False,
                                      replay_file=replay_file)

    def run(self):
        for payload in self.runner.measure(bg_shots=self.nbg, beam_shots=self.nbeam):
            self.payload_signal.emit(payload)
            print("Hello", payload.tds_percentage, payload.dispersion_setpoint, payload.is_bg)

from dataclasses import dataclass
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
        com, _ = transform_pixel_widths(com, np.zeros_like(com), pixel_units="m", dimension="y", to_variances=False)
        grad = np.gradient(com, phase)
        seconds_per_degree = TDS_PERIOD / 360
        grad2 = grad / seconds_per_degree
        return grad2

    def gradient2(self):
        phase, com = self.section2()
        com, _ = transform_pixel_widths(com, np.zeros_like(com), pixel_units="m", dimension="y", to_variances=False)
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




from .calibrate_tds import smooth, get_longest_two_monotonic_intervals, get_truncated_longest_sections

class TDSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = tds_window.Ui_MainWindow()
        self.ui.setupUi(self)
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.log_output_widget.append)

        self.machine = I1TDSCalibratingMachine("./")

        self.timer = QTimer()
        self.timer.timeout.connect(self.read_tds_from_machine)
        self.timer2hz = QTimer()
        self.timer2hz.timeout.connect(self.show_beam)
        self.timer.start(100)
        self.timer2hz.start(500)

        from IPython import embed

        # embed()
        self.ui.phase_spin_box_2.valueChanged.connect(self.machine.tds.set_phase)
        self.ui.amplitude_spin_box.valueChanged.connect(self.machine.tds.set_amplitude)
        self.ui.on_beam_push_button.clicked.connect(self.machine.tds.switch_on_beam)
        self.ui.off_beam_push_button.clicked.connect(self.machine.tds.switch_off_beam)
        self.ui.plus_180_phase.clicked.connect(lambda: self.machine.tds.set_phase(180 + self.machine.tds.read_sp_phase()))
        self.ui.minus_180_phase.clicked.connect(lambda: self.machine.tds.set_phase(-180 + self.machine.tds.read_sp_phase()))
        self.ui.start_voltage_calibration_button.clicked.connect(self.do_time_calibration)
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
        self.time_calibration_thread = TDSTimeCalibrationWorker(self.machine, self.get_amplitudes())
        self.time_calibration_thread.calibration_payload_signal.connect(self.update_time_calibration_plot)
        self.time_calibration_thread.start()

    def update_time_calibration_plot(self, payload):
        p, y = payload.raw_phase, payload.raw_com
        p1, y1 = payload.section1()
        p2, y2 = payload.section2()
        l1, = self.ui.voltage_calibration_plot.axes.plot(p, y, label=f"{payload.tds_amplitude}%, raw", linestyle="--", alpha=0.5)
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

        #self.img_plot.setLabel('left', "N bunch", units='')
        #self.img_plot.setLabel('bottom', "", units='eV')
        self.img_plot.hideAxis('left')
        self.img_plot.hideAxis('bottom')
        self.img = pg.ImageItem()

        self.img_plot.addItem(self.img)

        colormap = cm.get_cmap('viridis') #"nipy_spectral")  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt





        # Custom ROI for selecting an image region
        #self.roi = pg.ROI([20, 20], [6, 50])
        #self.roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        #self.roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        #self.img_plot.addItem(self.roi)
        #self.roi.setZValue(10)  # make sure ROI is drawn above image

        # Isocurve drawing
        #iso = pg.IsocurveItem(level=0.8, pen='g')
        #iso.setParentItem(self.img)
        #iso.setZValue(5)

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


        #hist.setLevels(data.min(), data.max())

        # build isocurves from smoothed data
        #iso.setData(pg.gaussianFilter(data, (2, 2)))

        # set position and scale of image
        #self.img.scale(0.2, 0.2)
        #self.img.translate(-50, 0)

        # zoom to fit imageo
        #self.img_plot.autoRange()
        # Apply the colormap
        self.img.setLookupTable(lut)
# class EnergySpreadMeasurementReplayMainWindow(QMainWindow)
