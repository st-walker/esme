import json
import sys
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from esme.control.screens import ScreenMetadata
from esme.gui.ui.slicer import Ui_slice_analysis_gui
from esme.gui.workers import ImagePayload


class XMultiYPlot(QWidget):
    def __init__(self, layout):
        super().__init__()
        # Create PlotWidget
        self.widget = PlotWidget()

        # Add the PlotWidget to the parent layout
        layout.addWidget(self.widget)

    def plot(
        self,
        df,
        x_data=None,
        x_label=None,
        y_data=None,
        y_axis=None,
        y_label=None,
        title=None,
        markers=False,
        legend=True,
        grid=True,
        colormap="glasbey",
    ):
        """
        Plots the data from the provided DataFrame.

        Parameters:
        df (pandas.DataFrame): The DataFrame containing the data to be plotted.
        x_data (str): The column name for the x-axis data.
        x_label (str): The label for the x-axis.
        y_data (list of str): The column names for the y-axis data.
        y_axis (list of str): The corresponding y-axes for the y_data. Each element should be either 'y1' or 'y2' to specify the left or right y-axis.
        y_label (list of str): The labels for the y-axes ["y1_label", "y2_label"].
        title (str): The title of the plot.
        markers (bool): Whether to display markers on the plot.
        legend (bool): Whether to display the legend.
        grid (bool): Whether to display the grid.
        colormap (str): The colormap to use for the plot.
        """
        # Clear the plot
        self.widget.clear()

        # Prepare the colormap
        cmap = pg.colormap.get(colormap, source="colorcet")

        # Create a new ViewBox for the right y-axis
        self.right_vb = pg.ViewBox()
        self.widget.scene().addItem(self.right_vb)
        self.widget.getAxis("right").linkToView(self.right_vb)
        self.right_vb.setXLink(self.widget.plotItem)

        # Show legend
        if legend:
            self.widget.addLegend()

        # Set the labels and title
        self.widget.setLabel("bottom", x_label)
        self.widget.setLabel("left", y_label[0])
        self.widget.getAxis("right").setLabel(y_label[1])
        self.widget.setTitle(title)

        # Set the grid and legend
        self.widget.showGrid(x=grid, y=grid)

        # Plot the data
        for i, col in enumerate(y_data):
            color = cmap[i]
            if y_axis[i] == "y1":
                self.widget.plot(
                    df[x_data].to_numpy(),
                    df[col].to_numpy(),
                    pen=pg.mkPen(color=color),
                    name=y_label[i],
                )
            else:
                self.right_vb.addItem(
                    pg.PlotCurveItem(
                        df[x_data].to_numpy(),
                        df[col].to_numpy(),
                        pen=pg.mkPen(color=color),
                        name=y_label[i],
                    )
                )

        # Adjust the view to fit the data
        self.widget.autoRange()

        # Connect the resizeEvent signal to the updateViews function
        self.widget.getViewBox().sigResized.connect(self.updateViews)

        # Show the right axis
        self.widget.getAxis("right").show()

    def updateViews(self):
        """
        Updates the views of the plot to keep the right y-axis in sync.
        """
        # Set the geometry of the right ViewBox to match the main ViewBox
        self.right_vb.setGeometry(self.widget.getViewBox().sceneBoundingRect())

        # Update the linked axes
        self.right_vb.linkedViewChanged(self.widget.getViewBox(), self.right_vb.XAxis)


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


class SliceAnalysisWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.ui = Ui_slice_analysis_gui()
        self.ui.setupUi(self)

        self._connect_buttons()
        self.centroid_plot, self.lps_plot = self._init_plots()

        self._image_payloads = None

        self.load_image_from_file()

    def _init_plots(self):
        centroid_plot = self.ui.centroid_profile_graphics.addPlot()
        centroid_plot.setLabel("bottom", "Time")
        centroid_plot.setLabel("left", "Position")
        lps_plot = self.ui.lps_graphics.addPlot()
        lps_plot.setLabel("bottom", "Time")
        lps_plot.setLabel(
            "left", "Slice Width"
        )  # Or slice energy spread OR beta OR ...

        return centroid_plot, lps_plot

    def _connect_buttons(self) -> None:
        self.ui.load_image_from_file_button.clicked.connect(self.load_image_from_file)

    def _get_image_payload(self) -> npt.NDArray[np.uint16] | None:
        if self._image_payloads:
            self._image_payloads[self.ui.pick_image_from_file_spinner.value()]
        return self._image_payloads[0]

    def update_ui(self) -> None:
        pass

    def _plot_beam_centroids(self) -> None:
        self.centroid_plot.clear()
        imp = self._get_image_payload()
        time, mu, sigma = imp.slice_gaussfit_time_calibrated
        mu_values = np.array([m.n for m in mu])
        mu_error = np.array([m.s for m in mu])
        error_bar_item = pg.ErrorBarItem(
            x=time, y=mu_values, top=mu_error, bottom=mu_error, beam=0.1
        )
        self.centroid_plot.addItem(error_bar_item)

    def _plot_lps(self) -> None:
        self.lps_plot.clear()
        imp = self._get_image_payload()
        time, mu, sigma = imp.slice_gaussfit_time_calibrated
        sigma_values = np.array([s.n for s in sigma])
        sigma_error = np.array([s.s for s in sigma])
        error_bar_item = pg.ErrorBarItem(
            x=time, y=sigma_values, top=sigma_error, bottom=sigma_error, beam=0.1
        )

        self.centroid_plot.addItem(error_bar_item)

    def _update_plots(self) -> None:
        self._plot_beamc_centroids()
        self._plot_lps()

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
        bunch_charge = md["bunch_charge"]

        data = pd.read_pickle(images_file)
        images = [d["data"] for d in data]

        if bg:
            for image in images:
                images -= bg
                image.clip(min=0, out=image)

        payloads = [
            ImagePayload(im, smd, time_calibration, bunch_charge, is_bg=False)
            for im in images
        ]

        self._image_payloads = payloads

        self._update_plots()


def start_slice_analysis_gui() -> None:
    app = QApplication(sys.argv)

    panel = SliceAnalysisWindow()
    panel.show()

    sys.exit(app.exec_())
