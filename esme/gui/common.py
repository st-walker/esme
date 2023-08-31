from dataclasses import dataclass
import yaml
import re
import socket
from pathlib import Path
from importlib_resources import files
import logging
import pyqtgraph as pg
from matplotlib import cm
import numpy as np

from PyQt5.QtCore import QObject, pyqtSignal, QByteArray, QBuffer, QIODevice
from PyQt5.QtWidgets import QWidget

from esme.control.configs import build_simple_machine_from_config, load_virtual_machine_interface, build_lps_machine_from_config

from esme.control.vmint import DictionaryXFELMachineInterface


DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"
DEFAULT_VCONFIG_PATH = files("esme.gui") / "vmachine.yaml"


def is_in_controlroom():
    name = socket.gethostname()
    reg = re.compile(r"xfelbkr[0-9]\.desy\.de")
    return bool(reg.match(name))


def build_default_machine_interface():
    mi = None
    if not is_in_controlroom():
        with open(DEFAULT_VCONFIG_PATH, "r") as f:
            doocsdict = yaml.safe_load(f)
            mi = load_virtual_machine_interface(doocsdict)
    machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH, mi=mi)
    return machine

def build_default_lps_machine():
    mi = None
    if not is_in_controlroom():
        with open(DEFAULT_VCONFIG_PATH, "r") as f:
            doocsdict = yaml.safe_load(f)
            mi = load_virtual_machine_interface(doocsdict)
    machine = build_lps_machine_from_config(DEFAULT_CONFIG_PATH, mi=mi)
    return machine

def get_default_virtual_machine_interface():
    with open(DEFAULT_VCONFIG_PATH, "r") as f:
        doocsdict = yaml.safe_load(f)
        return load_virtual_machine_interface(doocsdict)

def get_config_path():
    return Path.home() / ".config" / "diagnostics-utility/"

def get_i1_calibration_config_dir():
    return get_config_path() / "i1-tds-calibrations"

class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


def setup_screen_display_widget(widget):
    image_plot = widget.addPlot()
    image_plot.clear()
    image_plot.hideAxis("left")
    image_plot.hideAxis("bottom")
    image = pg.ImageItem(autoDownsample=True, border="k")

    image_plot.addItem(image)

    colormap = cm.get_cmap("viridis")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)

    image.setLookupTable(lut)
    # print(lut.shape)

    return image_plot



def load_scanner_panel_ui_defaults():
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        dconf = yaml.safe_load(f)
        uiconf = dconf["scanner"]["gui_defaults"]

        return uiconf

        # if is_in_controlroom():
        #     outdir = Path(uiconf["outdir_bkr"])
        # else:
        #     outdir = Path("./measurements")

        # return ScanSettings(quad_wait=quad_wait,
        #                     tds_amplitude_wait=tds_amplitude_wait,
        #                     beam_on_wait=beam_on_wait,
        #                     outdir=outdir
        
def get_screenshot(window_widget):
    screenshot_tmp = QByteArray()
    screeshot_buffer = QBuffer(screenshot_tmp)
    screeshot_buffer.open(QtCore.QIODevice.WriteOnly)
    widget = QWidget.grab(window_widget)
    widget.save(screeshot_buffer, "png")
    return screenshot_tmp.toBase64().data().decode()
