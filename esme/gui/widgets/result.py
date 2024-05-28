import logging
from collections import defaultdict
from pathlib import Path
from textwrap import dedent

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QTableWidgetItem, QVBoxLayout, QWidget

from esme.analysis import DerivedBeamParameters, OpticsFixedPoints
from esme.gui.ui import Ui_results_box_dialog
from esme.gui.widgets.common import df_to_logbook_table, send_to_logbook
from esme.plot import formatted_parameter_dfs

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class ScannerResultsDialog(QtWidgets.QDialog):
    DEFAULT_AUTHOR = "High Resolution Slice Energy Measurer"
    DEFAULT_TITLE = "Slice Energy Spread @ OTRC.64.I1D"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_results_box_dialog()
        self.ui.setupUi(self)

        self.ui.send_to_logbook_button.clicked.connect(self.send_to_logbook)
        self.ui.close_button.clicked.connect(self.close)
        self.measurement_parameters = None
        self.skip_timestamps = set()
        self.plot_items_to_timestamps = defaultdict(dict)

    def send_to_logbook(self):
        # TODO: maybe also send screenshot of parent panel?  if it exists...s
        text = self.ui.comments_browser.toPlainText()

        fitted_beam_parameters = self.measurement_parameters
        fit_df, beam_df = formatted_parameter_dfs(fitted_beam_parameters)
        fit_string = df_to_logbook_table(fit_df)
        beam_string = df_to_logbook_table(beam_df)

        text = dedent(
            f"""\
        {text}

        !!Derived Beam Parameters
        {beam_string}

        !!Fit Parameters
        {fit_string}
        """
        )

        # # C
        # with (Path(self.online_measurement_result) / "notes.txt").open("w") as f:
        #     f.write(text)

        # shutil.copy(DEFAULT_CONFIG_PATH, self.online_measurement_result)

        send_to_logbook(
            title=self.DEFAULT_TITLE,
            author=self.DEFAULT_AUTHOR,
            severity="MEASURE",
            text=text,
        )

    def post_measurement_result(
        self,
        measurement_parameters: DerivedBeamParameters,
        output_directory: Path = None,
    ):
        message = ""
        if output_directory is not None:
            message = f"Written files to {output_directory}\n\n"

        self.ui.comments_browser.setPlainText(message)
        self.ui.title_line_edit.setText(self.DEFAULT_TITLE)

        self.measurement_parameters = measurement_parameters

        fit_df, beam_df = formatted_parameter_dfs(measurement_parameters)

        self.fill_beam_table(beam_df, fit_df)
        self.ui.comments_browser.setFocus()

    def fill_beam_table(self, df, df2):
        # Space either side of units is a hardcoded hack so that the
        # mm.mrad units are visible in the column...
        # Yes this method is hideous.
        header = ["Variable", "Values", "Alt. Values", "     Units     "]

        row_labels = [
            "<i>σ</i><sub>E</sub>",
            "<i>σ</i><sub>I</sub>",
            "<i>σ</i><sub>E</sub><sup>TDS</sup>",
            "<i>σ</i><sub>B</sub>",
            "<i>σ</i><sub>R</sub>",
            "<i>ε</i><sub><i>x</i></sub>",
            "<i>σ</i><sub><i>z</i></sub><sup>Gaus.</sup>",
            "<i>σ</i><sub><i>t</i></sub><sup>Gaus.</sup>",
            "<i>σ</i><sub><i>z</i></sub><sup>RMS</sup>",
            "<i>σ</i><sub><i>t</i></sub><sup>RMS</sup>",
        ]

        row_units = ["keV", "μm", "keV", "μm", "μm", "mm·mrad", "mm", "ps", "mm", "ps"]

        bt = self.ui.beam_parameters_table
        bt.setColumnCount(len(header))
        bt.setRowCount(len(row_labels) + len(df2))
        for i, tup in enumerate(df.itertuples()):
            label = row_labels[i]
            value = tup.values
            alt_value = tup.alt_values
            units = row_units[i]

            set_richtext_widget(bt, i, 3, units)

            widget = QtWidgets.QWidget()
            widgetText = QtWidgets.QLabel(label)
            widgetText.setWordWrap(True)
            widgetLayout = QtWidgets.QHBoxLayout()
            widgetLayout.addWidget(widgetText)
            widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)

            widget.setLayout(widgetLayout)

            bt.setCellWidget(i, 0, widget)
            if "nan" not in value:
                value_item = QTableWidgetItem(value)
                bt.setItem(i, 1, value_item)
            if "nan" not in alt_value:
                alt_value_item = QTableWidgetItem(alt_value)
                bt.setItem(i, 2, alt_value_item)

            # units_item = QTableWidgetItem(units)
            # bt.setItem(i, 3, units_item)

        labels2 = {
            "V_0": "<i>V</i><sub>0</sub>",
            "D_0": "<i>D</i><sub>0</sub>",
            "E_0": "<i>E</i><sub>0</sub>",
            "A_V": "<i>A</i><sub><i>V</i></sub>",
            "B_V": "<i>B</i><sub><i>V</i></sub>",
            "A_D": "<i>A</i><sub><i>D</i></sub>",
            "B_D": "<i>B</i><sub><i>D</i></sub>",
            "A_beta": "<i>A</i><sub><i>β</i></sub>",
            "B_beta": "<i>B</i><sub><i>β</i></sub>",
        }

        units2 = {
            "V_0": "MV",
            "D_0": "m",
            "E_0": "MeV",
            "A_V": "m<sup>2</sup>",
            "B_V": "m<sup>2</sup>V<sup>-2</sup>",
            "A_D": "m<sup>2</sup>",
            "B_D": "",
            "A_beta": "m<sup>2</sup>",
            "B_beta": "m",
        }

        for i, tup in enumerate(df2.itertuples(), start=i + 1):
            label = labels2[tup.Index]

            value = tup.values
            units = units2[tup.Index]

            set_richtext_widget(bt, i, 0, label)

            value_item = QTableWidgetItem(value)
            bt.setItem(i, 1, value_item)
            set_richtext_widget(bt, i, 3, units)

        bt.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustToContents)

        bt.setHorizontalHeaderLabels(header)
        bt.resizeRowsToContents()
        bt.resizeColumnsToContents()


class WidthsPlotWidget(QWidget):
    timestamp_signal = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.init_ui()
        self.skip_timestamps = set()
        self.plot_items_to_timestamps = defaultdict(dict)

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        self.widths_layout = pg.GraphicsLayoutWidget()

        # Plot labels
        labels_common = dict(left=["Widths", "px"])
        dispersion_labels = dict(bottom=["Dispersion", "m"]) | labels_common
        tds_labels = dict(bottom=["Voltage", "V"]) | labels_common
        beta_labels = dict(bottom=["Beta", "m"]) | labels_common

        # Add plots to the GraphicsLayoutWidget and make them easily
        # accessibly by making them members of the class

        self.dscan_widths_plot = self.create_plot(
            "Dispersion Scan", "Dispersion", "Widths", xunits="m", yunits="px"
        )
        self.widths_layout.nextColumn()
        self.tscan_widths_plot = self.create_plot(
            "Voltage Scan", "Voltage", "Widths", xunits="V", yunits="px"
        )
        self.widths_layout.nextColumn()
        self.bscan_widths_plot = self.create_plot(
            "Beta Scan", "Beta", "Widths", xunits="m", yunits="px"
        )

        # Widgets to layout
        main_layout.addWidget(self.widths_layout)

        # Slider layout
        slider_layout = QHBoxLayout()
        slider_label = QLabel("Image Processing Step")
        slider_layout.addWidget(slider_label)
        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        slider_layout.addWidget(self.slider)
        main_layout.addLayout(slider_layout)

        self.setLayout(main_layout)

    def create_plot(self, title, xlabel, ylabel, xunits=None, yunits=None):
        plotDataItem = pg.PlotDataItem(pen=None, symbol="o")
        plotDataItem.sigPointsClicked.connect(self.on_point_clicked)
        plot = self.widths_layout.addPlot(title=title)
        plot.addItem(plotDataItem)
        plot.setLabel("bottom", xlabel, units=xunits)
        plot.setLabel("left", ylabel, units=yunits)
        return plotDataItem

    def on_point_clicked(self, plot, spots):
        # Always just take the first point, it is meaningless to emit
        # multiple points in this application...

        spot = spots[0]
        point = spot.pos()

        x = point.x()
        y = point.y()

        timestamp = self.plot_items_to_timestamps[plot][(x, y)]
        self.timestamp_signal.emit(timestamp)

    def get_plot_data(self, plot_item):
        # Example of accessing data from a plot item. Adjust according to your data structure.
        # This is a placeholder to demonstrate the concept.
        if plot_item.listDataItems():
            data_item = plot_item.listDataItems()[0]
            return data_item.xData, data_item.yData
        return np.array([]), np.array([])  # Return empty arrays if no data

    def _plot_scan(self, xvar, scan, plot_item):
        x = []
        y = []
        ts = []

        LOG.info(f"Plotting scan with {self.skip_timestamps=}")

        for setpoint_x, setpoint in zip(xvar, scan.setpoints):
            widths = setpoint.get_final_widths(skip_timestamps=self.skip_timestamps)
            timestamps = setpoint.timestamps(skip_timestamps=self.skip_timestamps)
            widths = [w.n for w in widths]

            y.extend(widths)
            x.extend(np.ones_like(widths) * setpoint_x)
            ts.extend(timestamps)

        # Store plot+point+timestamp map for emitting when points are clicked
        for x_, y_, t in zip(x, y, ts):
            print(x_, y_)
            self.plot_items_to_timestamps[plot_item][(x_, y_)] = t

        plot_item.setData(
            x,
            y,
            pen=None,
            symbol="o",
            symbolPen=None,
            symbolSize=10,
            symbolBrush=(100, 100, 255, 200),
        )

    def plot_measurement(self, measurement, avmapping):
        dispersion = measurement.dscan.dispersions()
        amplitudes = measurement.tscan.amplitudes()
        voltages = avmapping(amplitudes)
        beta = measurement.bscan.betas()

        # Reset this dictionary so we don't just forever keep adding to it.
        self.plot_items_to_timestamps = defaultdict(dict)

        self._plot_scan(dispersion, measurement.dscan, self.dscan_widths_plot)
        self._plot_scan(voltages, measurement.tscan, self.tscan_widths_plot)
        self._plot_scan(beta, measurement.bscan, self.bscan_widths_plot)


class FullResultWidget(QWidget):
    timestamp_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.ofp = OpticsFixedPoints(beta_screen=0.6, beta_tds=4.3, alpha_tds=1.9)
        self.skip_timestamps = set()
        self.avmapping = None

    def set_skip_timestamps(self, timestamps: set):
        LOG.info(f"Regenerating measurement result with: {timestamps}")
        self.skip_timestamps |= timestamps
        self.widths_plot.skip_timestamps = timestamps
        self.post_measurement()

    def set_keep_timestamps(self, timestamps: set):
        self.skip_timestamps -= timestamps
        LOG.info(f"Regenerating measurement result without: {timestamps}")
        self.widths_plot.keep_timestamps = timestamps
        self.post_measurement()

    def update_avmapping(self, avmapping):
        self.avmapping = avmapping
        self.post_measurement()

    def init_ui(self):
        # Create a horizontal layout
        layout = QHBoxLayout()

        self.result_dialog = ScannerResultsDialog()
        self.widths_plot = WidthsPlotWidget()

        # Just reemit the timestamp signal from the widths widget.
        self.widths_plot.timestamp_signal.connect(self.timestamp_signal.emit)

        # Create two widgets to be displayed side by side
        left_widget = self.result_dialog
        right_widget = self.widths_plot

        # Add the widgets to the layout
        layout.addWidget(left_widget)
        layout.addWidget(right_widget)

        # Set the layout on the application's main widget
        self.setLayout(layout)

    def post_measurement(self, measurement=None):
        if measurement is not None:
            self.measurement = measurement

        if self.avmapping is None:
            return

        result = self.measurement.get_derived_beam_parameters(
            slice_width=3,
            skip_timestamps=self.skip_timestamps,
            avmapping=self.avmapping,
        )

        try:
            self.result_dialog.post_measurement_result(result)
        except ValueError:
            from IPython import embed

            embed()

        self.widths_plot.plot_measurement(self.measurement, avmapping=self.avmapping)


def set_richtext_widget(table, row, column, text):
    widget = QtWidgets.QWidget()
    widgetText = QtWidgets.QLabel(text)
    widgetText.setWordWrap(True)
    widgetLayout = QtWidgets.QHBoxLayout()
    widgetLayout.addWidget(widgetText)
    widgetLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
    widget.setLayout(widgetLayout)

    table.setCellWidget(row, column, widget)
