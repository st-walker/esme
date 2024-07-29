import base64
import io
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from scipy.constants import c, e, m_e

from esme.control.screens import ScreenMetadata
from esme.gui.ui.slicer import Ui_slice_analysis_gui
from esme.gui.widgets.common import send_to_logbook
from esme.gui.workers import ImagePayload

EMASS_EV = m_e * c**2 / e


@dataclass
class SliceMeasurementParameters:
    screen_name: str
    tpixel_size: float
    epixel_size: float
    time_calibration: float | None = None
    energy_calibration: float | None = None
    voltage: float | None = None
    beta: float | None = None
    beam_energy: float | None = None
    screen_resolution: float | None = None


class InjectorSliceEnergySpreadCalculator:
    def __init__(
        self,
        screen_resolution=0,
        energy=130,
        beta=0,
        emitn=0,
        dispersion=0,
        tds_voltage=0,
    ):
        self.sigma_r = screen_resolution
        # Assume that whatever energy we are at, we are on energy...  so energy = energy0...
        self.energy = energy  # eV
        self.energy0 = energy
        self.beta = beta  # twiss beta at screen
        self.emitn = emitn  # normalised emittance
        self.dispersion = 0.0  # dispersion at screen
        self.tds_voltage = tds_voltage  # MV
        # constants:
        self.tds_wavenumber = 2 * np.pi * 3e9 / c  # 3 GHz
        self.tds_length = 0.7
        self.tds_twiss0y = (
            1.881059,
            4.334833,
            1.046957,
        )  # alpha, beta, gamma at stard of TDS

    def relgamma0(self):
        return self.energy0 / EMASS_EV

    def sigma_m(self, sigma_e):
        # sigma_m in the plot is in mm, so
        return np.sqrt(
            (
                self._screen_resolution_contribution2()
                + self._betatron_contribution2()
                + self._dispersive_contribution2(sigma_e)
                + self.tds_contribution2()
            )
        )

    def _screen_resolution_contribution2(self):
        return self.sigma_r**2

    def _betatron_contribution2(self):
        return self.sigma_b2() * self.energy0 / self.energy

    def _dispersive_contribution2(self, sigma_e) -> float:
        return (self.dispersion * sigma_e / self.energy) ** 2

    def tds_contribution2(self):
        D = self.dispersion
        k = self.tds_wavenumber
        V = self.tds_voltage
        E = self.energy
        E0 = self.energy0
        sigma_i2 = self.sigma_i() ** 2
        return (D * e * k * V) ** 2 * E0 * sigma_i2 / E**3

    def sigma_i(self):
        alpha, beta, gamma = self.tds_twiss0y
        l = self.tds_length
        return (
            self.emitn * (beta + 0.25 * l**2 * gamma - l * alpha) / self.relgamma0()
        )

    def sigma_b2(self):
        return (self.beta * self.emitn / self.relgamma0()) ** 2

    def sigma_e(self, sigma_m):
        sigma_m = sigma_m * 1e-3  # Convert mm to metres.
        sigma_e = (
            np.sqrt(
                sigma_m**2
                - self._screen_resolution_contribution2()
                - self._betatron_contribution2()
                - self.tds_contribution2()
            )
            * self.energy
            / abs(self.dispersion)
        )
        return sigma_e


class SliceAnalysisWindow(QWidget):
    def __init__(self, parent=None, image_worker=None):
        super().__init__(parent)

        self.ui = Ui_slice_analysis_gui()
        self.ui.setupUi(self)

        self._connect_buttons()
        self._init_spinners()

        self.i1calc = InjectorSliceEnergySpreadCalculator()
        (
            self.width_ax,
            self.centroid_ax,
            self.image_ax,
            self._espread_ax,
        ) = self._init_plots()
        self.image_worker = image_worker
        self.ui.new_image_button.setEnabled(bool(self.image_worker))
        self.ui.append_image_button.setEnabled(bool(self.image_worker))

        self.logger_dialogue = LogbookDialog()

        self._image_payloads = []
        # Store references to the lines we are plotting, keys = image indices
        self._lps_lines = {}
        self._centroid_lines = {}

        # self.load_image_from_file()

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_ui)
        self.timer.start(1000)

    def _init_spinners(self):
        self.ui.beam_energy_spinner.setValue(130)  # MeV.

    def _redraw_canvas(self):
        self.ui.phase_space_canvas.canvas.draw()

    def _update_ui(self):
        self._update_n_images_stored_label()
        self._update_espread_calculator_from_ui()
        self.ui.new_image_button.setEnabled(bool(self.image_worker))
        self.ui.append_image_button.setEnabled(bool(self.image_worker))

    def _set_espread_ui_enabled(self, *, enable=False):
        self.ui.screen_resolution_spinner.setEnabled(enable)
        self.ui.sigma_r_label.setEnabled(enable)
        self.ui.voltage_label.setEnabled(enable)
        self.ui.voltage_spinbox.setEnabled(enable)
        self.ui.beta_label.setEnabled(enable)
        self.ui.beta_spinbox.setEnabled(enable)
        self.ui.energy_label.setEnabled(enable)
        self.ui.beam_energy_spinner.setEnabled(enable)
        self.ui.emittance_label.setEnabled(enable)
        self.ui.emittance_spinner.setEnabled(enable)

    def _update_n_images_stored_label(self) -> None:
        try:
            nstored = len(self._image_payloads)
        except TypeError:
            nstored = 0
        self.ui.images_stored_label.setText(f"Images Stored: {nstored}")

    def _update_espread_calculator_from_ui(self):
        c = self.i1calc
        ui = self.ui
        c.sigma_r = ui.screen_resolution_spinner.value() * 1e-6  # µm to m
        c.tds_voltage = ui.voltage_spinbox.value() * 1e6  # MV to V
        c.beta = ui.beta_spinbox.value()
        c.dispersion = ui.dispersion_spinner.value()
        c.energy = ui.beam_energy_spinner.value() * 1e6  # MeV to eV
        c.energy0 = c.energy
        c.emitn = ui.emittance_spinner.value() * 1e-6  # mm.mrad to m.rad
        self._redraw_canvas()

    def _init_plots(self):
        fig = self.ui.phase_space_canvas.fig
        width_ax = fig.add_subplot(311)
        centroid_ax = fig.add_subplot(312, sharex=width_ax)
        image_ax = fig.add_subplot(313, sharex=width_ax)
        espread_ax = None
        centroid_ax.set_ylabel("Position / µm")
        image_ax.set_ylabel("Position / mm")
        width_ax.set_ylabel("Slice Width / mm")
        image_ax.set_xlabel("Time / ps")

        return width_ax, centroid_ax, image_ax, espread_ax

    def _connect_buttons(self) -> None:
        self.ui.load_image_from_file_button.clicked.connect(self.load_image_from_file)
        self.ui.clear_plots_button.clicked.connect(self._clear_plots)
        self.ui.plot_selected_image_button.clicked.connect(
            self._add_selected_image_to_plots
        )
        self.ui.display_image_only_button.clicked.connect(self._update_image)
        self.ui.dispersion_spinner.valueChanged.connect(self._set_dispersion)
        self.ui.image_number_spinner.valueChanged.connect(self._select_image_number)
        self.ui.update_label_button.clicked.connect(self._update_label)
        self.ui.label_line_edit.returnPressed.connect(self._update_label)
        self.ui.plot_all_button.clicked.connect(self.plot_all)
        self.ui.remove_selected_image_button.clicked.connect(
            self._remove_selected_image_from_plots
        )
        self.ui.new_image_button.clicked.connect(self.get_new_image_from_screen)
        self.ui.append_image_button.clicked.connect(self.append_image_from_screen)
        self.ui.clear_cache_button.clicked.connect(self._clear_image_cache)
        self.ui.send_to_logbook_button.clicked.connect(
            self._open_send_to_logbook_dialog
        )
        self.ui.cancel_button.clicked.connect(self.close)

    def _open_send_to_logbook_dialog(self) -> None:
        text = textwrap.dedent(
            f"""\
        |Screen| {self.ui.screen_name_label.text()}
        |Time Calibration| {self.ui.time_cal_spinner.value()} us/ps
        |Energy Calibration| {self.ui.energy_cal_spinner.value()} MeV/m
        """
        )

        if dispersion := self.ui.dispersion_spinner.value():
            text += textwrap.dedent(
                f"""\
            |Dispersion| {dispersion} m"
            |Screen Resolution| {self.ui.screen_resolution_spinner.value()} µm
            |Voltage| {self.ui.voltage_spinbox.value()} MV
            |Beta| {self.ui.beta_spinbox.value()}
            |Beam Energy| {self.ui.beam_energy_spinner.value()} MeV
            |Emittance| {self.ui.emittance_spinner.value()} mm.mrad
            |TDS Contribution| {np.sqrt(self.i1calc.tds_contribution2())} m
            |Betatron Contribution| {np.sqrt(self.i1calc._betatron_contribution2())} m
            """
            )

        self.logger_dialogue.show_as_new_modal_dialogue(
            text_to_append=text, figure=self.ui.phase_space_canvas.fig
        )

    def _clear_image_cache(self):
        self._image_payloads = []
        self._clear_plots()
        self._update_n_images_stored_label()
        self.ui.label_line_edit.setText("")

    def _remove_selected_image_from_plots(self):
        imp, index = self._get_image_payload()
        try:
            self._lps_lines[index].remove()
        except KeyError:
            pass
        else:
            del self._lps_lines[index]

        try:
            self._centroid_lines[index].remove()
        except KeyError:
            pass
        else:
            del self._centroid_lines[index]

        try:
            self.width_ax.legend().legend()
        except AttributeError:
            pass

    def _select_image_number(self, image_number: int) -> None:
        maxlen = max(0, len(self._image_payloads) - 1)
        image_number = np.clip(image_number, 0, maxlen)
        self.ui.image_number_spinner.setValue(image_number)

        try:
            line = self._lps_lines[image_number]
        except:
            text = ""
        else:
            text = line.get_label()
        self.ui.label_line_edit.setText(text)

    def _set_dispersion(self, dispersion: float) -> None:
        if not dispersion and self._espread_ax:
            self._espread_ax.remove()
            self._espread_ax = None
            self._update_espread_calculator_from_ui()
        elif dispersion and not self._espread_ax:
            self._update_espread_calculator_from_ui()
            self._espread_ax = self.width_ax.secondary_yaxis(
                "right", functions=(self.i1calc.sigma_e, self.i1calc.sigma_m)
            )
            self._espread_ax.set_ylabel(r"$\sigma_E$ / eV")

        self._set_espread_ui_enabled(enable=bool(dispersion))

    def _clear_plots(self):
        _espread_line_artists = []
        if self._espread_ax:
            _espread_line_artists = self._espread_ax.lines
            self._espread_ax.set_prop_cycle(None)
        all_artists = (
            self.width_ax.lines
            + self.centroid_ax.lines
            + self.image_ax.images
            + _espread_line_artists
        )
        for artist in all_artists:
            artist.remove()

        # Remove legend if there is one...
        try:
            self.width_ax.legend_.remove()
        except AttributeError:
            pass
        # Reset colour cycles.
        self.centroid_ax.set_prop_cycle(None)
        self.width_ax.set_prop_cycle(None)
        self.ui.phase_space_canvas.canvas.draw()

    def _get_image_payload(self) -> tuple[ImagePayload, int]:
        index = self.ui.image_number_spinner.value()
        return self._image_payloads[index], index

    def _plot_beam_centroids(self, imp, index) -> None:
        if index in self._centroid_lines:  # Don't plot twice
            return
        time, mu, sigma = imp.slice_gaussfit_time_calibrated
        mu_values = np.array([m.n for m in mu])
        (line,) = self.centroid_ax.plot(time * 1e12, mu_values * 1e3)
        self._centroid_lines[index] = line

    def _plot_lps(self, imp, index) -> None:
        if index in self._lps_lines:  # Don't plot twice
            return
        # self._clear_ax_plot_contents(self.width_ax)
        time, mu, sigma = imp.slice_gaussfit_time_calibrated
        sigma_values = np.array([s.n for s in sigma])
        label = f"Image {index}"
        (line,) = self.width_ax.plot(time * 1e12, sigma_values * 1e6, label=label)
        # Store reference to the line based on its index.
        self.ui.label_line_edit.setText(label)
        self._lps_lines[index] = line
        self.width_ax.legend()

    def _update_label(self) -> None:
        label = self.ui.label_line_edit.text()
        _, index = self._get_image_payload()
        self._lps_lines[index].set_label(label)
        self.width_ax.legend()
        self._redraw_canvas()

    def plot_all(self) -> None:
        self._clear_plots()
        for index, payload in enumerate(self._image_payloads):
            self._plot_lps(payload, index)
            self._plot_beam_centroids(payload, index)
        # Only plot the last image because each overwrites the previous anyway.
        self._plot_image(payload, index)
        self.ui.phase_space_canvas.canvas.draw()

    def _add_selected_image_to_plots(self) -> None:
        imp, index = self._get_image_payload()
        self._plot_beam_centroids(imp, index)
        self._plot_lps(imp, index)
        self._plot_image(imp, index)
        self.ui.phase_space_canvas.canvas.draw()

    def _update_image(self) -> None:
        for artist in self.image_ax.images:
            artist.remove()
        imp, index = self._get_image_payload()
        self._plot_image(imp, index)
        self.ui.phase_space_canvas.canvas.draw()

    def _plot_image(self, imp, index) -> None:
        image = imp.image
        time = imp.time_calibrated * 1e12
        # This could/should be calibrated to ENERGY and DISTANCE at some point...
        dr = imp.energy * 1e3
        self.image_ax.imshow(
            image, aspect="auto", extent=[min(time), max(time), min(dr), max(dr)]
        )
        for artist in self.image_ax.texts:
            artist.remove()
        self.image_ax.text(
            max(time) * 0.7, max(dr) * 0.7, f"Image {index}", color="white", fontsize=15
        )

    def get_new_image_from_screen(self) -> None:
        self._clear_plots()
        self.append_image_from_screen()

    def append_image_from_screen(self) -> None:
        imp = self._get_image_from_worker()
        self._image_payloads.append(imp)
        index = len(self._image_payloads) - 1
        self._plot_beam_centroids(imp, index)
        self._plot_lps(imp, index)
        self._plot_image(imp, index)
        self.ui.phase_space_canvas.canvas.draw()
        self.ui.image_number_spinner.setValue(index)

    def _get_image_from_worker(self) -> None:
        imq = self.image_worker.subscribe(nimages=1)
        imp = imq.q.get(block=True, timeout=3)
        return imp

    def load_image_from_file(self) -> None:
        dir_name = QFileDialog.getExistingDirectory(
            self,
            "Load Measurement Directory",
            "",
        )
        # dir_name = Path("/Users/stuartwalker/Downloads/test-beam")

        # Also need to look next to this file for the image metadata, e.g.
        # pixel size, time calibration, screen name, etc..
        if not dir_name:
            return

        images_file = Path(dir_name) / "beam_images.pkl"

        bg_file = Path(dir_name) / "bg_images.pkl"
        if bg_file.exists():
            bg_data = pd.read_pickle(bg_file)
            bg_images = bg_data["data"]
            bg = bg_images.mean(axis=0)
        else:
            bg = 0

        metadata_file = Path(dir_name) / "snapshot.json"
        with metadata_file.open("r") as f:
            md = json.load(f)

        # streaking_plane = ?
        smd = ScreenMetadata(
            xsize=md["xpixel_size"],
            ysize=md["ypixel_size"],
            nx=md["nxpixels"],
            ny=md["nypixels"],
        )
        time_calibration = md["time_calibration"]
        energy_calibration = md["energy_calibration"]
        bunch_charge = md["bunch_charge"]

        data = pd.read_pickle(images_file)
        images = [d["data"] for d in data]

        self.ui.screen_name_label.setText(md["screen"])
        self.ui.time_cal_spinner.setValue(time_calibration * 1e-6)
        if energy_calibration:
            self.ui.time_cal_spinner.setValue(energy_calibration)

        if bg:
            for image in images:
                images -= bg
                image.clip(min=0, out=image)
        payloads = [
            ImagePayload(im, smd, time_calibration, bunch_charge, is_bg=False)
            for im in images
        ]
        self._image_payloads = payloads
        self._add_selected_image_to_plots()
        self._update_n_images_stored_label()


class LogbookDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
        self.text_append = ""
        self.figure = None

    def _init_ui(self) -> None:
        self.setWindowTitle("Logbook Entry")
        # Create the Author label and line edit
        author_label = QLabel("Author:", self)
        self.author_edit = QLineEdit(self)
        self.author_edit.setPlaceholderText("xfeloper")

        author_layout = QHBoxLayout()
        author_layout.addWidget(author_label)
        author_layout.addWidget(self.author_edit)

        # Create the main text edit
        self.text_edit = QTextEdit(self)
        self.text_edit.setAcceptRichText(False)
        self.text_edit.setPlaceholderText("Logbook Entry...")

        # Create the buttons
        self.send_button = QPushButton("Send to XFEL e-Logbook", self)
        self.cancel_button = QPushButton("Cancel", self)

        self.send_button.clicked.connect(self.send_to_logbook)
        self.cancel_button.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.cancel_button)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addLayout(author_layout)
        layout.addWidget(self.text_edit)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        # Set initial focus
        if not self.author_edit.text():
            self.author_edit.setFocus()
        else:
            self.text_edit.setFocus()

    def send_to_logbook(self) -> None:
        # Implement your logic to send the content to the XFEL e-Logbook here
        text = self.text_edit.toPlainText()
        author = self.author_edit.text() or "xfeloper"
        title = "Longitudinal Phase Space via the TDS"
        severity = "MEASURE"
        full_text = f"{text}\n----\n{self.text_append}"

        with io.BytesIO() as bytes_io:
            self.figure.savefig(bytes_io, format="png")
            bytes_io.seek(0)
            png_string = base64.b64encode(bytes_io.read()).decode("utf-8")

        send_to_logbook(
            author=author,
            title=title,
            severity=severity,
            text=full_text,
            image=png_string,
        )
        self.accept()

    def show_as_new_modal_dialogue(
        self, text_to_append: str, figure: plt.Figure
    ) -> None:
        self.text_to_append = text_to_append
        self.figure = figure
        self.setModal(True)
        self.show()


def start_slice_analysis_gui() -> None:
    app = QApplication(sys.argv)

    panel = SliceAnalysisWindow()
    panel.show()

    sys.exit(app.exec_())
