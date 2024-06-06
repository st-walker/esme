import sys
import logging
import re
import socket
import subprocess
import sys
import traceback
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyqtgraph as pg
import yaml
import matplotlib
from PyQt5.QtCore import QBuffer, QByteArray, QIODevice, QObject, pyqtSignal, QPoint, pyqtProperty
from PyQt5.QtWidgets import QMessageBox, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QPixmap, QColor, QPainter, QPolygon

from esme import DiagnosticRegion
from esme.control.configs import (
    MachineManagerFactory,
    build_area_watcher_from_config,
    load_virtual_machine_interface,
)
from esme.control.dint import DOOCSInterface, DOOCSInterfaceABC
from esme.control.vdint import DictionaryDOOCSInterface

try:
    pass
except ImportError:
    pass
else:
    pg.setConfigOption("useNumba", True)

DEFAULT_CONFIG_PATH = files("esme.gui.widgets") / "defaultconf.yaml"
DEFAULT_VCONFIG_PATH = files("esme.gui.widgets") / "vmachine.yaml"

_MACHINE_MANAGER_FACTORY: MachineManagerFactory | None = None


def get_machine_manager_factory() -> MachineManagerFactory:
    global _MACHINE_MANAGER_FACTORY
    if _MACHINE_MANAGER_FACTORY:
        return _MACHINE_MANAGER_FACTORY
    di = make_default_doocs_interface()
    _MACHINE_MANAGER_FACTORY = MachineManagerFactory(
        DEFAULT_CONFIG_PATH, default_dint=di
    )
    return _MACHINE_MANAGER_FACTORY


def is_in_controlroom() -> bool:
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
    return build_area_watcher_from_config(
        DEFAULT_CONFIG_PATH, area=DiagnosticRegion.I1, di=di
    )


def make_b2_watcher():
    di = make_default_doocs_interface()
    return build_area_watcher_from_config(
        DEFAULT_CONFIG_PATH, area=DiagnosticRegion.B2, di=di
    )


def get_default_virtual_machine_interface() -> DictionaryDOOCSInterface:
    with open(DEFAULT_VCONFIG_PATH, "r") as f:
        doocsdict = yaml.safe_load(f)
        return load_virtual_machine_interface(doocsdict)


def get_config_path() -> Path:
    return Path.home() / ".config" / "lps/"


def get_tds_calibration_config_dir() -> Path:
    return get_config_path() / "tds"


def set_machine_by_region(widget: QWidget, location: DiagnosticRegion) -> None:
    if location == "I1":
        widget.machine = widget.i1machine
    elif location == "B2":
        widget.machine = widget.b2machine
    else:
        raise ValueError(f"Unkonwn location string: {location}")


def set_tds_calibration_by_region(
    widget: QWidget, calibration: DiagnosticRegion
) -> None:
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


def setup_screen_display_widget(widget: pg.GraphicsLayoutWidget, axes: bool = False, units: str = "m") -> pg.PlotItem:
    # We add a plot to the pg.GraphicsLayoutWidget  
    main_plot = widget.addPlot()
    # Clear it in case it has something in it already.
    main_plot.clear()
    main_plot.setAspectLocked(True)
    # Row-major is apparently faster.
    image = pg.ImageItem(border="k", axisOrder="row-major")
    main_plot.hideAxis("left")
    main_plot.hideAxis("bottom")
    if axes:
        main_plot.setLabel("bottom", "<i>&Delta;x</i>", units=units)
        main_plot.setLabel("right", "<i>&Delta;y</i>", units=units)
    # add the image item to the plot
    main_plot.addItem(image)
    cmap = matplotlib.colormaps["viridis"]
    cmap._init()
    lut = (cmap._lut * (255)).view(np.ndarray)
    image.setLookupTable(lut)
    return main_plot


def load_scanner_panel_ui_defaults() -> dict[str, Any]:
    with open(DEFAULT_CONFIG_PATH, "r") as f:
        dconf = yaml.safe_load(f)
        uiconf = dconf["scanner"]["gui_defaults"]
        return uiconf


def get_screenshot(widget: QWidget) -> str:
    # Create byte array with buffer for writing to the array and open
    # it.
    screenshot_tmp = QByteArray()
    screenshot_buffer = QBuffer(screenshot_tmp)
    screenshot_buffer.open(QIODevice.WriteOnly)
    # Get QPixmap of the widget
    pixmap = QWidget.grab(widget)
    # Write the pixmap to the byte array via the buffer as png.
    pixmap.save(screenshot_buffer, "png")
    # convert bytes to to base64 (i.e. printable characters)
    # Then get the underlying bytes object, but we want
    # a str, so we call decode on it.
    return screenshot_tmp.toBase64().data().decode()


def send_to_logbook(
    author: str = "",
    title: str = "",
    severity: str = "",
    text: str = "",
    image: str | None = None,
) -> None:
    """
    Send information to the electronic logbook.

    """

    # The DOOCS elog expects an XML string in a particular format. This string
    # is being generated in the following as an initial list of strings.
    elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', "<entry>"]

    # author information
    elogXMLStringList.extend(
        [
            "<author>",
            author,
            "</author>",
            "<title>",
            title,
            "</title>",
            "<severity>",
            severity,
            "</severity>",
            "<text>",
            text,
            "</text>",
        ]
    )
    # image information
    if image is not None:
        elogXMLStringList.extend(["<image>", image, "</image>"])

    elogXMLStringList.append("</entry>")
    # join list to the final string
    elogXMLString = "\n".join(elogXMLStringList)
    lpr = subprocess.Popen(
        ["/usr/bin/lp", "-o", "raw", "-d", "xfellog"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    # send printer job
    lpr.communicate(elogXMLString.encode("utf-8"))


def send_widget_to_log(
    widget: QWidget,
    author: str = "",
    title: str = "",
    severity: str = "",
    text: str = "",
) -> None:
    send_to_logbook(
        author=author,
        title=title,
        severity=severity,
        text=text,
        image=get_screenshot(widget),
    )


def df_to_logbook_table(df: pd.DataFrame) -> str:
    """Given a pandas DataFrame instance,
    return a string of the table formatted correctly for writing in the EuXFEL log book.
    """
    table_string = df.to_csv(sep="|", lineterminator="\n|")
    table_string = table_string[:-1]
    return table_string


def raise_message_box(
    text: str, *, informative_text: str, title: str, icon: str = "NoIcon"
) -> None:
    """icon: one of {None, "NoIcon", "Question", "Information", "Warning", "Critical"}
    text: something simple like "Error", "Missing config"
    informative_text: the actual detailed message that you read
    title: the window title

    """

    msg = QMessageBox()

    icon = {
        None: QMessageBox.NoIcon,
        "noicon": QMessageBox.NoIcon,
        "question": QMessageBox.Question,
        "information": QMessageBox.Information,
        "warning": QMessageBox.Warning,
        "critical": QMessageBox.Critical,
    }[icon.lower()]

    msg.setIcon(icon)
    msg.setText(text)
    msg.setInformativeText(informative_text)
    msg.setWindowTitle(title)
    msg.exec_()


def make_exception_hook(program_name: str) -> Callable[[], None]:
    def hook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        error_message = "".join(
            traceback.format_exception(exc_type, exc_value, exc_traceback)
        )
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle("Unhandled Exception")

        msg_box.setText(
            f"An uncaught except has been raised.  {program_name} will exit.  Error:"
        )
        msg_box.setInformativeText(error_message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
        sys.exit(1)

    return hook


class PlayPauseButton(QPushButton):
    play_signal = pyqtSignal()
    pause_signal = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)

        self._play_icon = self.create_play_icon()
        self._pause_icon = self.create_pause_icon()

        self.clicked.connect(self.toggle_action)
        self.is_playing = False

    @pyqtProperty(bool)
    def is_playing(self) -> bool:
        return self._is_playing
    
    @is_playing.setter
    def is_playing(self, value: bool) -> None:
        self._is_playing = bool(value)
        if value:
            self.play_signal.emit()
        else:
            self.pause_signal.emit()
        self._update_button_icon()

    @staticmethod
    def create_play_icon() -> QIcon:
        pixmap = QPixmap(50, 50)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw play triangle
        painter.setBrush(QColor(0, 0, 0))  # Black color for contrast
        triangle = QPolygon([QPoint(15, 10), QPoint(35, 25), QPoint(15, 40)])
        painter.drawPolygon(triangle)
        painter.end()

        return QIcon(pixmap)

    @staticmethod
    def create_pause_icon() -> QIcon:
        pixmap = QPixmap(50, 50)
        pixmap.fill(QColor(0, 0, 0, 0))  # Transparent background
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        # Draw pause bars
        painter.setBrush(QColor(0, 0, 0))  # Black color for contrast
        painter.drawRect(12, 10, 8, 30)
        painter.drawRect(30, 10, 8, 30)
        painter.end()

        return QIcon(pixmap)

    def _update_button_icon(self) -> None:
        # This method should only set the visuals,
        # and should not emit anything.
        if self.is_playing:
            self.setIcon(self._pause_icon)
            self.setText("Pause")
        else:
            self.setIcon(self._play_icon)
            self.setText("Play")

    def toggle_action(self) -> None:
        self.is_playing = not self.is_playing
