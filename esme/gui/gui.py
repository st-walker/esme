from pathlib import Path
import os
import sys
from matplotlib import cm
import numpy as np
import socket
import re
from esme.inout import i1_dscan_config_from_scan_config_file, i1_tds_amplitudes_from_scan_config_file


from PyQt5.QtWidgets import QFrame, QMainWindow, QApplication
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer

from esme.measurement import MeasurementPayload, ScanType
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


def start_tds_gui(scantoml, debug_mode, replay):
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


class TDSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = esme_window.Ui_MainWindow()
        self.ui.setupUi(self)
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.measurement_log_browser.append)
    

# class EnergySpreadMeasurementReplayMainWindow(QMainWindow)
