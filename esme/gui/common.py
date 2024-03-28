import yaml
import re
import socket
from pathlib import Path
from importlib.resources import files
import logging
import pyqtgraph as pg
from matplotlib import cm
import numpy as np
import subprocess

from PyQt5.QtCore import QObject, pyqtSignal, QByteArray, QBuffer, QIODevice
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QMessageBox

from esme.control.configs import (load_virtual_machine_interface,
                                    build_lps_machine_from_config,
                                    make_hires_injector_energy_spread_machine, 
                                    build_area_watcher_from_config)
from esme import DiagnosticRegion

from esme.control.sbunches import SpecialBunchesControl
from esme.control.dint import DOOCSInterfaceABC, DOOCSInterface
from esme.control.vdint import DictionaryDOOCSInterface

try:
    import numba
except ImportError:
    pass
else:
    pg.setConfigOption('useNumba', True)



DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yaml"
DEFAULT_VCONFIG_PATH = files("esme.gui") / "vmachine.yaml"


def is_in_controlroom():
    name = socket.gethostname()
    reg = re.compile(r"xfelbkr[0-9]\.desy\.de")
    return bool(reg.match(name))

def make_default_doocs_interface() -> DOOCSInterfaceABC:
    if not is_in_controlroom():
        return get_default_virtual_machine_interface()
    else:
        return DOOCSInterface()

def make_i1_watcher():
    di = make_default_doocs_interface()
    return build_area_watcher_from_config(DEFAULT_CONFIG_PATH, area=DiagnosticRegion.I1, di=di)

def make_b2_watcher():
    di = make_default_doocs_interface()
    return build_area_watcher_from_config(DEFAULT_CONFIG_PATH, area=DiagnosticRegion.B2, di=di)

def make_default_injector_espread_machine():
    di = make_default_doocs_interface()
    return make_hires_injector_energy_spread_machine(DEFAULT_CONFIG_PATH, di=di)

    
def make_default_i1_lps_machine():
    di = make_default_doocs_interface()
    return build_lps_machine_from_config(DEFAULT_CONFIG_PATH, DiagnosticRegion("I1"), di=di)

def make_default_b2_lps_machine():
    di = make_default_doocs_interface()
    return build_lps_machine_from_config(DEFAULT_CONFIG_PATH, DiagnosticRegion("B2"), di=di)

def get_default_virtual_machine_interface() -> DictionaryDOOCSInterface:
    with open(DEFAULT_VCONFIG_PATH, "r") as f:
        doocsdict = yaml.safe_load(f)
        return load_virtual_machine_interface(doocsdict)

def make_default_sbm(location=None):
    di = make_default_doocs_interface()
    if location is None:
        location = DiagnosticRegion.I1
    return SpecialBunchesControl(location=location, di=di)

def get_config_path():
    return Path.home() / ".config" / "lps/"

def get_tds_calibration_config_dir():
    return get_config_path() / "tds"

def set_machine_by_region(widget, location: DiagnosticRegion):
    if location == "I1":
        widget.machine = widget.i1machine
    elif location == "B2":
        widget.machine = widget.b2machine
    else:
        raise ValueError(f"Unkonwn location string: {location}")

def set_tds_calibration_by_region(widget, calibration: DiagnosticRegion):
    if not hasattr(widget, "i1machine") and not hasattr(widget, "b2machine"):
        raise TypeError("Widget missing i1machine or b2machine instances")

    location = calibration.region
    if location == "I1":
        widget.i1machine.deflector.calibration = calibration
    elif location == "B2":
        widget.b2machine.deflector.calibration = calibration
    else:
        raise ValueError(f"Unkonwn location string: {location}")
    


class QPlainTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)

    def emit(self, record):
        msg = self.format(record)
        self.log_signal.emit(msg)


def setup_screen_display_widget(widget, axes=False, units="m"):
    image_plot = widget.addPlot()
    image_plot.clear()
    image = pg.ImageItem(border="k")
    # if not axes:
    image_plot.hideAxis("left")
    image_plot.hideAxis("bottom")
    if axes:
        image_plot.setLabel("bottom", "<i>&Delta;x</i>", units=units)
        image_plot.setLabel("right", "<i>&Delta;y</i>", units=units)    
    
    image_plot.addItem(image)

    colormap = cm.get_cmap("viridis")
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray)

    image.setLookupTable(lut)
    print(lut.shape)

    
    return image_plot



def load_scanner_panel_ui_defaults():
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        dconf = yaml.safe_load(f)
        uiconf = dconf["scanner"]["gui_defaults"]
        return uiconf

def get_screenshot(window_widget):
    screenshot_tmp = QByteArray()
    screeshot_buffer = QBuffer(screenshot_tmp)
    screeshot_buffer.open(QIODevice.WriteOnly)
    widget = QWidget.grab(window_widget)
    widget.save(screeshot_buffert, "png")
    return screenshot_tmp.toBase64().data().decode()


def send_to_logbook(author="", title="", severity="", text="", image=None) -> None:
    """
    Send information to the electronic logbook.

    """

    # The DOOCS elog expects an XML string in a particular format. This string
    # is beeing generated in the following as an initial list of strings.
    elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

    # author information
    elogXMLStringList.append('<author>')
    elogXMLStringList.append(author)
    elogXMLStringList.append('</author>')
    # title information
    elogXMLStringList.append('<title>')
    elogXMLStringList.append(title)
    elogXMLStringList.append('</title>')
    # severity information
    elogXMLStringList.append('<severity>')
    elogXMLStringList.append(severity)
    elogXMLStringList.append('</severity>')
    # text information
    elogXMLStringList.append('<text>')
    elogXMLStringList.append(text)
    elogXMLStringList.append('</text>')
    # image information
    if image is not None:
        encodedImage = base64.b64encode(image)
        elogXMLStringList.append('<image>')
        elogXMLStringList.append(encodedImage.decode())
        elogXMLStringList.append('</image>')
    elogXMLStringList.append('</entry>')
    # join list to the final string
    elogXMLString = '\n'.join(elogXMLStringList)
    print(elogXMLStringList)
    with open("logbook-stuart-attempt.xml", "w") as f:
        f.write(elogXMLString)
    # open printer process
    print(elogXMLString)
    elog = "xfellog"
    lpr = subprocess.Popen(
        ['/usr/bin/lp', '-o', 'raw', '-d', elog],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    # send printer job
    lpr.communicate(elogXMLString.encode('utf-8'))

def send_widget_to_log(widget, author="", title="", severity="", text=""):
    image = get_screenshot(widget)
    send_to_logbook(author="", title="", severity="", text="", image=image)


def df_to_logbook_table(df):
    table_string = df.to_csv(sep="|", lineterminator="\n|")
    table_string = table_string[:-1]
    return table_string
    

def raise_message_box(text, *, informative_text, title, icon=None):
    """icon: one of {None, "NoIcon", "Question", "Information", "Warning", "Critical"}
    text: something simple like "Error", "Missing config"
    informative_text: the actual detailed message that you read
    title: the window title

    """

    msg = QMessageBox()

    icon = {None: QMessageBox.NoIcon,
            "NoIcon": QMessageBox.NoIcon,
            "Question": QMessageBox.Question,
            "Information": QMessageBox.Information,
            "Warning": QMessageBox.Warning,
            "Critical": QMessageBox.Critical}[icon]

    msg.setIcon(icon)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.setWindowTitle(title)
    msg.exec_()
