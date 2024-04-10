"""Class providing a GUI for exploring the results of a measurement.

# This file is part of the ESME project.

The data model is as usual, at the top level a DecomposedMeasurement,
which contains 3 lists of DecomposedSetpoint instances (dscan, bscan
and tscan), which in turn contain a list of DecomposedBeamImage
instances.

The idea being the affix "Decomposed" is that we cache all our
intermediate image processing results so we can really see what is
going on in the analysis.



"""


import os
import numpy as np
import sys
import datetime
import logging
# import crash_ipdb  # just import crash_ipdb in your Python code
from typing import Optional
import time
import pandas as pd

from esme.gui.ui.explorer import Ui_MainWindow
from esme.load import load_result_directory, load_calibration_from_result_directory
from esme.analysis import DecomposedBeamImage, make_outlier_widths_mask, MeasurementDataFrames, SliceWidthsFitter, OpticsFixedPoints, DerivedBeamParameters
from esme.image import make_image_processing_pipeline
from esme.gui.widgets.common import setup_screen_display_widget
from esme.gui.widgets.scannerpanel import ScanType
from esme.gui.widgets.result import FullResultWidget
from esme.gui.tds_calibrator import CalibrationExplorer
from esme.analysis import true_bunch_length_from_setpoint
from esme.calibration import AmplitudeVoltageMapping

from PyQt5.QtWidgets import QApplication, QComboBox, QLabel, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QPushButton)
from PyQt5.QtCore import QAbstractItemModel, QModelIndex, Qt, pyqtSignal
import pyqtgraph as pg
from enum import Enum, auto


pg.setConfigOptions(antialias=True)



LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


# @dataclass
# class AnalysisConfig:
#     skip_timestamps: Optional[set] = None
#     keep_timestamps: Optional[set] = None
#     slice_width: int = 3
#     sigma_cut: int = 3


class MaskState(Enum):
    Masked = auto()
    NotMasked = auto()
    PartiallyMasked = auto()


def start_explorer_gui(initial_dir: Optional[os.PathLike] = None,
                       calibration_file: Optional[os.PathLike] = None) -> None:
    """
    Launches a GUI application for data exploration.

    This function initializes and displays the main window of a data explorer
    application, using PyQt5 for the graphical user interface. It takes optional
    parameters to specify the initial result directory to be loaded upon launch and
    an optional TDS calibration file for the TDS calibration.

    Parameters:
    - initial_dir (Optional[str]): The path to the measurement result directory
      that the application should display upon startup.
    - calibration_file (Optional[str]): The path to a TDS calibration file.
      If none then it gets them from the scan.yaml file.

    Raises:
    - SystemExit: This function calls sys.exit() which raises a SystemExit
      exception upon the application's exit.

    Example:
    ```
    start_explorer_gui('/path/to/result/dir', '/path/to/calibration/file.yaml')
    ```
    """
    app = QApplication(sys.argv)

    main_window = DataExplorer(initial_dir=initial_dir)

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


class DataExplorer(QMainWindow):
    skip_timestamps_signal = pyqtSignal(set)
    keep_timestamps_signal = pyqtSignal(set)
    calibration_signal = pyqtSignal(object)

    def __init__(self, initial_dir=None, calibration_file=None):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        if initial_dir is not None:
            self.initialise_tree(initial_dir)
            tds_calibration = load_calibration_from_result_directory(initial_dir, calibration_file)

        self.ui.tree_view.activated.connect(self.click_row)

        self.image_plot = setup_screen_display_widget(self.ui.glwidget, axes=True, units="px")

        self.xplot = None
        self.yplot = None
        self.current_image = None

        self.ui.improc_slider.setEnabled(False)
        self.ui.improc_slider_label.setEnabled(False)
        self.ui.improc_slider.valueChanged.connect(self.update_image_from_slider)

        self.ui.slice_ana_checkbox.stateChanged.connect(self.set_slice_ana_subplots)
        self.set_slice_ana_subplots(self.ui.slice_ana_checkbox.checkState())

        self.ui.calc_ses_button.clicked.connect(self.calculate_slice_energy_spread)

        self.result_window = FullResultWidget()

        self.connect_timestamp_signals()

        self.ui.breakdown_image_ana_button.clicked.connect(self.breakdown_image_analysis)
        # self.ui.breakdown_image_ana_button.click()

        self.ui.tree_view.expandAll()

        self.get_decomposed_measurement()
        self.cali_window = CalibrationExplorer(ccalib=tds_calibration,
                                               fname=initial_dir,
                                               parent=self)
        self.cali_window.avmapping_signal.connect(self.update_amplitude_voltage_mapping)
        self.ui.tds_calib_button.clicked.connect(self.cali_window.show)
        self.update_amplitude_voltage_mapping(self.cali_window.get_amplitude_voltage_mapping())

        header = self.ui.tree_view.header()
        header.setSectionResizeMode(0, header.ResizeToContents)
        header.setSectionResizeMode(1, header.ResizeToContents)
        header.setSectionResizeMode(2, header.ResizeToContents)
        header.setSectionResizeMode(3, header.ResizeToContents)

    def connect_timestamp_signals(self):
        self.result_window.timestamp_signal.connect(self.handle_timestamp_signal)
        # Timestamp masking signals
        self.skip_timestamps_signal.connect(self.result_window.set_skip_timestamps)
        self.keep_timestamps_signal.connect(self.result_window.set_keep_timestamps)
        self.skip_timestamps_signal.connect(self.ui.tree_view.model().handle_skip_timestamps)
        self.ui.tree_view.model().skip_timestamps_signal.connect(self.skip_timestamps_signal)
        self.ui.tree_view.model().keep_timestamps_signal.connect(self.keep_timestamps_signal)
        


    def update_amplitude_voltage_mapping(self, avmapping):
        self.calibration_signal.emit(avmapping)
        self.ui.tree_view.model().update_voltages(avmapping)
        self.result_window.avmapping = avmapping

    def calculate_slice_energy_spread(self) -> None:
        sigma_cut = self.ui.outlier_sigma_cut_spinner.value()
        slice_width = self.ui.slice_width_spinner.value()

        # Get the decomp_measurement and post it to the result window,
        # then force regenerate the result and show.
        decomp_measurement = self.ui.tree_view.model().get_decomposed_measurement()

        # Emit any masked timestamps here.
        if self.ui.mask_outliers_checkbox.isChecked():
            skip_timestamps = decomp_measurement.get_outlier_timestamps(slice_width=slice_width, sigma_cut=sigma_cut)
            self.skip_timestamps_signal.emit(set(skip_timestamps))

        self.result_window.post_measurement(decomp_measurement)
        self.result_window.show()

    def should_plot_slice_analysis(self):
        return self.ui.slice_ana_checkbox.isChecked()

    def set_slice_ana_subplots(self, value):
        win = self.ui.glwidget
        if value == Qt.Unchecked:
            win.clear()
            self.image_plot = setup_screen_display_widget(win, axes=True, units="px")
        elif value == Qt.Checked:
            self.yplot = win.addPlot(row=0, col=1)
            win.nextRow()
            self.xplot = win.addPlot(row=1, col=0)
            self.yplot.setMaximumWidth(200)
            self.xplot.setMaximumHeight(200)
            self.xplot.setLabel("bottom", "Column Index (Cropped)")
            # self.xplot.setLabel("left", "Intensity")
            self.xplot.setXLink(self.image_plot)
            self.yplot.setYLink(self.image_plot)
            self.yplot.setLabel("bottom", "<i>&sigma;<sub>M</sub></i>")

    def get_decomposed_measurement(self):
        return self.ui.tree_view.model().get_decomposed_measurement()

    def breakdown_image_analysis(self):
        decomp_measurement = self.get_decomposed_measurement()
        self.image_processing_window = ImageProcessingStepImpactWidget(decomp_measurement)
        self.image_processing_window.selected_timestamp.connect(self.handle_timestamp_signal)

        self.image_processing_window.show()
        self.image_processing_window.raise_()

    def update_image_from_slider(self, slider_value):
        selected_rows = self.ui.tree_view.selectionModel().selectedRows()
        assert len(selected_rows) == 1
        model = self.ui.tree_view.model()
        item = model.get_item(selected_rows[0])
        dbim = item.dbim
        image = dbim.get_image_at_step(slider_value)

        if self.crop_image() and False:
            crop = dbim.final_cropping_slice
            image = image[crop]

        print(image.shape)

        self.post_beam_image(image, title=model.image_titles[slider_value])
        self.post_slice_analysis(dbim, image_processing_step=slider_value)

    def click_row(self, index: QModelIndex) -> None:
        item = index.internalPointer()
        name = item.name

        if name == "Image":
            self._handle_image_row(index)

    def _model_index_from_timestamp(self, timestamp):
        index = self.ui.tree_view.model().index_from_timestamp(timestamp)
        return index

    def handle_timestamp_signal(self, timestamp: float):
        index = self._model_index_from_timestamp(timestamp)

        self.ui.tree_view.scrollTo(QModelIndex(index))
        self.ui.tree_view.setCurrentIndex(QModelIndex(index))
        # Treat the jump to this index just as if it had been clicked on by the user)
        self.click_row(QModelIndex(index))

    def _handle_image_row(self, index: QModelIndex) -> None:
        item = index.internalPointer()
        model = index.model()
        dbim = item.dbim

        self.ui.improc_slider.setEnabled(True)
        self.ui.improc_slider_label.setEnabled(True)

        raw_image = dbim.im0
        if self.crop_image() and False:
            crop = dbim.final_cropping_slice
            raw_image = raw_image[crop]

        self.ui.improc_slider.setValue(0)
        self.ui.improc_slider.setMinimum(0)
        self.ui.improc_slider.setMaximum(dbim.pipeline_steps)

        self.post_beam_image(raw_image, title=model.image_titles[0])
        self.post_slice_analysis(dbim, image_processing_step=0)

    def post_slice_analysis(self, dbim, image_processing_step):
        if self.should_plot_slice_analysis():
            rows, means, sigmas = dbim.get_slice_properties_at_step(image_processing_step)
            index = dbim.max_energy_row_index()
            image = dbim.get_image_at_step(image_processing_step)
            if False:
                image = image[dbim.final_cropping_slice]
            self.xplot.clear()
            self.xplot.plot(image[index])
            self.yplot.clear()
            self.yplot.plot([s.n for s in sigmas], np.arange(len(rows)))

            # XXX: Cheekily assume first item is the image item... maybe probelmatic in the future.
            for item in self.image_plot.items[1:]:
                self.image_plot.removeItem(item)

            self.image_plot.plot([m.n for m in means], rows)
            self.image_plot.addLine(y=index)

            self.yplot.addLine(y=index)

    def crop_image(self):
        return self.ui.crop_checkbox.isChecked()

    def slice_ana(self):
        return self.ui.slice_ana_checkbox.isChecked()

    def post_beam_image(self, image, title=""):
        items = self.image_plot.items
        image_item = items[0]
        image_item.setImage(image.T, levelSamples=len(image.flatten()))
        self.ui.improc_step_label.setText(title)
        # self.image_plot.setTitle(title)

    def initialise_tree(self, dirname):
        measurement = load_result_directory(dirname)
        model = DataExplorerModel(measurement)
        self.ui.tree_view.setModel(model)


class DecomposedSetpoint:
    """A setpoint containts multiple images at a particular machine
    configuration (optics + TDS setting) with methods for dealing with
    decomposed results.

    """
    def __init__(self, images: list[DecomposedBeamImage],
                 amplitude: float,
                 dispersion: float,
                 beta: float):
        self.images = images
        self.amplitude: float = amplitude
        self.dispersion: float = dispersion
        self.beta: float = beta

    def timestamps(self, skip_timestamps: Optional[set] = None) -> np.ndarray:
        if skip_timestamps is None:
            skip_timestamps = set()

        timestamps = np.array([a.tagged_raw_image.metadata.kvps["timestamp"] for a in self.images])
        return np.array([t for t in timestamps if t not in skip_timestamps])

    @property
    def df(self):
        return pd.DataFrame([image.tagged_raw_image.metadata.kvps for image in self.images])

    def energies(self):
        return np.array([image.tagged_raw_image.energy for image in self.images])

    @property
    def screen_name(self):
        return self.images[0].tagged_raw_image.metadata.screen_name

    def _get_dbims_with_timestamps(self) -> list[DecomposedBeamImage]:
        yield from zip(self.images, self.timestamps())

    def get_final_widths(self, slice_width: int = 3, skip_timestamps: Optional[set] = None) -> np.ndarray:
        final_widths = []
        assert slice_width >= 1

        # This is how we mask images ultimately, we have timestamps that we skip here.
        if skip_timestamps is None:
            skip_timestamps = set()
        else:
            skip_timestamps = set(skip_timestamps)

        for dbim, timestamp in self._get_dbims_with_timestamps():
            if timestamp in skip_timestamps:
                print("Skipping: ", timestamp)
                continue

            _, means, sigmas = dbim.final_slice_analysis()
            row_index = means.argmin()

            if slice_width == 1:
                import ipdb; ipdb.set_trace()

            if slice_width == 1:
                sigmas = [sigmas[row_index]]
            else:
                slice_low = row_index - slice_width // 2
                slice_high = row_index + (slice_width + 1) // 2
                sigmas = sigmas[slice_low: slice_high]

            final_widths.append(np.mean(sigmas))

            if np.isnan(final_widths[-1].n):
                import ipdb; ipdb.set_trace()



        return final_widths

    def get_outlier_timestamps(self, slice_width: int, sigma_cut: int = 3) -> np.ndarray:
        widths_with_errors = self.get_final_widths(slice_width=slice_width)
        try:
            widths = [x.n for x in widths_with_errors]
        except:
            import ipdb; ipdb.set_trace()
        return self.timestamps()[make_outlier_widths_mask(widths, sigma_cut=sigma_cut)]

    def get_all_widths_at_all_stages(self) -> np.ndarray:
        all_images_widths_pipeline_stages = []
        # Iterating over all images here basically
        for dbim in self.images:
            # We get the index and then apply it retroactively to all
            # the steps of the image processing, this ensures we are
            # consistent in our slice selection.
            chosen_row_index = dbim.max_energy_row_index()
            pipeline_stage_widths = []
            # Iterating over all pipeline steps for the image.
            for i in range(dbim.pipeline_steps + 1):
                _, _, sigmas = dbim.get_slice_properties_at_step(i)
                chosen_sigma = sigmas[chosen_row_index].n
                pipeline_stage_widths.append(chosen_sigma)

            all_images_widths_pipeline_stages.append(pipeline_stage_widths)


        all_images_all_pipeline_stages_with_outlier_filtering = make_array_with_final_outlier_filtering_step(
            all_images_widths_pipeline_stages
        )

        return all_images_all_pipeline_stages_with_outlier_filtering

    def __repr__(self) -> str:
        a = self.amplitude
        d = self.dispersion
        return f"<{type(self).__name__}, A={a}%, D={d}m, β={self.beta}m>"


class DecomposedScan:
    def __init__(self, scan_type: ScanType, setpoints: list[DecomposedSetpoint]):
        self.scan_type = scan_type
        self.setpoints = setpoints

    def get_widths(self, slice_width=3, skip_timestamps=None) -> np.ndarray:
        """Get the final mean slice widths for each setpoint in the scan."""
        widths = []
        for sp in self.setpoints:
            widths.append(np.mean(sp.get_final_widths(slice_width=slice_width,
                                                      skip_timestamps=skip_timestamps)))
        return np.array(widths)

    def dispersions(self):
        return np.array([x.dispersion for x in self.setpoints])

    def amplitudes(self):
        return np.array([x.amplitude for x in self.setpoints])

    def betas(self):
        return np.array([x.beta for x in self.setpoints])


class DecomposedMeasurement:
    """
    Represents a top-level class for decomposing an entire measurement into its constituent setpoints for detailed analysis.

    This class is designed to handle and operate on decomposed measurements divided into dscan, tscan, and bscan components,
    allowing for the extraction and fitting of derived beam parameters based on the provided decomposed setpoints.

    Attributes:
        dscan (list[DecomposedSetpoint], optional): A list of decomposed setpoints for the dispersion scan.
        tscan (list[DecomposedSetpoint], optional): A list of decomposed setpoints for the tds scan.
        bscan (list[DecomposedSetpoint], optional): A list of decomposed setpoints for the beta scan.


    """

    def __init__(self, dscan: list[DecomposedSetpoint]=None,
                 tscan: list[DecomposedSetpoint]=None ,
                 bscan: list[DecomposedSetpoint]=None,
                 avmapping: Optional[AmplitudeVoltageMapping]=None):
        self.dscan = dscan
        self.tscan = tscan
        self.bscan = bscan if bscan else None
        self.avmapping = avmapping

    def get_derived_beam_parameters(self, slice_width=3, skip_timestamps=None, avmapping=None) -> DerivedBeamParameters:
        dispersions = [x.dispersion for x in self.dscan.setpoints]
        amplitudes = [x.amplitude for x in self.tscan.setpoints]
        betas = [x.beta for x in self.bscan.setpoints]

        dscan_widths = self.dscan.get_widths(slice_width=slice_width, skip_timestamps=skip_timestamps)
        tscan_widths = self.tscan.get_widths(slice_width=slice_width, skip_timestamps=skip_timestamps)

        try:
            bscan_widths = self.bscan.get_widths(slice_width=slice_width, skip_timestamps=skip_timestamps)
        except AttributeError:
            bscan_widths = None
        else:
            bscan_widths = dict(zip(betas, bscan_widths))


        sigma_z_gaussian = self.get_sigma_z_gaussian(avmapping=avmapping)
        sigma_z_rms = self.get_sigma_z_rms(avmapping=avmapping)

        fitter = SliceWidthsFitter(dict(zip(dispersions, dscan_widths)),
                                   dict(zip(amplitudes, tscan_widths)), bscan_widths,
                                   avmapping=avmapping)

        return fitter.all_fit_parameters(beam_energy=self.beam_energy(),
                                         dscan_voltage=self.dscan_voltage(),
                                         tscan_dispersion=self.tscan_dispersion(),
                                         optics_fixed_points=self.optics_fixed_points(),
                                         sigma_z=sigma_z_gaussian,
                                         sigma_z_rms=sigma_z_rms)

    def get_outlier_timestamps(self, slice_width,sigma_cut: int = 3) -> np.ndarray:
        result = []
        for sp in self.dscan.setpoints:
            result.extend(sp.get_outlier_timestamps(slice_width, sigma_cut=sigma_cut))
        for sp in self.tscan.setpoints:
            result.extend(sp.get_outlier_timestamps(slice_width, sigma_cut=sigma_cut))
        for sp in self.bscan.setpoints:
            result.extend(sp.get_outlier_timestamps(slice_width, sigma_cut=sigma_cut))

        return np.array(result)

    def _check_avmapping(self, avmapping: AmplitudeVoltageMapping) -> AmplitudeVoltageMapping:
        if avmapping is None and self.avmapping is None:
            raise ValueError("No avmapping for mapping TDS amplitudes to physical voltages has been supplied.")
        elif avmapping is None:
            return self.avmapping
        return avmapping

    def get_sigma_z_gaussian(self, avmapping: Optional[AmplitudeVoltageMapping] = None) -> tuple[float, float]:
        avmapping = self._check_avmapping(avmapping)
        index = np.argmax(self.tscan.amplitudes())
        max_voltage_sp = self.tscan.setpoints[index]
        bl = true_bunch_length_from_setpoint(max_voltage_sp, avmapping, method="gaussian")
        return bl.n, bl.s

    def get_sigma_z_rms(self, avmapping: Optional[AmplitudeVoltageMapping] = None) -> tuple[float, float]:
        avmapping = self._check_avmapping(avmapping)
        index = np.argmax(self.tscan.amplitudes())
        max_voltage_sp = self.tscan.setpoints[index]
        bl = true_bunch_length_from_setpoint(max_voltage_sp, avmapping, method="rms")
        return bl.n, bl.s

    def dscan_dispersions(self):
        return np.array([x.dispersion for x in self.dscan.setpoints])

    def tscan_voltages(self, avmapping):
        return np.array([x.voltage for x in self.tscan.setpoints])

    def tscan_amplitudes(self):
        return np.array([x.amplitude for x in self.tscan.setpoints])

    def bscan_betas(self):
        return np.array([x.beta for x in self.bscan.setpoints])

    def beam_energy(self) -> float:
        # in eV
        return 130e6

    def tscan_dispersion(self) -> float:
        return 1.2

    def dscan_voltage(self) -> float:
        return 610000.0

    def optics_fixed_points(self):
        ofp = OpticsFixedPoints(beta_screen=0.6, beta_tds=4.3, alpha_tds=1.9)
        return ofp

    # def energy(self):
    #     return 13
    #     energies = []
    #     energies.extend([t.energy for t in self.tscans])
    #     energies.extend([b.energy for b in self.bscans])
    #     energies.extend([d.energy for d in self.dscans])
    #     return np.mean(energies)


def make_array_with_final_outlier_filtering_step(all_images_all_pipeline_steps_widths: np.ndarray) -> np.ndarray:
    """This takes the an array of 2d widths, where each row
    corresponds to an each, and each column corresponds to an image
    processing step (right is later in the image processing).  The
    final step of the image processing is to remove outlier widths.
    Here we do this, we take each row and append the last element of
    that row if it is NOT an outlier, or np.nan if it IS an outlier.

    So e.g. [[1, 2], [3, 4]], if we select 4 to be an outlier for some reason, we end up with
    [[1, 2, 2],
    [3, 4, np.nan]].

    """

    all_images_all_pipeline_steps_widths = np.array(all_images_all_pipeline_steps_widths)
    outliers_mask = make_outlier_widths_mask(all_images_all_pipeline_steps_widths[..., -1],
                                             sigma_cut=3)

    return append_nans_to_rows_where(all_images_all_pipeline_steps_widths, outliers_mask)


def append_nans_to_rows_where(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Step 1: Identify the shape of the original array
    array.shape

    # Step 2: Create a new column based on the mask condition
    # If the mask is True, append np.nan, otherwise repeat the last column
    new_column = np.where(mask[:, None], np.nan, array[:, -1:])

    # Step 3: Append the new column to the original array
    modified_array = np.hstack((array, new_column))

    return modified_array


class TreeItem:
    def __init__(self, data: dict, parent: 'TreeItem' = None):
        # Instead of data method, it should be a dictionary, in which everything not set, is set to None.
        # This is a class where the zeroth column of the row is
        # responsible for all elements on its row.
        # self.item_data = column_entries
        self.item_data = data
        self.parent_item = parent
        self.child_items = []

    def child(self, number: int) -> 'TreeItem':
        if number < 0 or number >= len(self.child_items):
            return None
        return self.child_items[number]

    def flags(self, column):
        return Qt.ItemIsEnabled

    def is_column_checkable(self, column):
        return False

    def last_child(self):
        return self.child_items[-1] if self.child_items else None

    def child_count(self) -> int:
        return len(self.child_items)

    def child_number(self) -> int:
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

    def column_count(self) -> int:
        return len(self.item_data)

    def data(self, column: int, role: int = None):
        if column < 0 or column >= len(self.item_data):
            return None

        return self.item_data[column]

    def insert_children(self, position: int, count: int, columns: int) -> bool:
        if position < 0 or position > len(self.child_items):
            return False

        for row in range(count):
            data = [None] * columns
            item = TreeItem(data.copy(), self)
            self.child_items.insert(position, item)

        return True

    def insert_columns(self, position: int, columns: int) -> bool:
        if position < 0 or position > len(self.item_data):
            return False

        for column in range(columns):
            self.item_data.insert(position, None)

        for child in self.child_items:
            child.insert_columns(position, columns)

        return True

    def parent(self):
        return self.parent_item

    def remove_children(self, position: int, count: int) -> bool:
        if position < 0 or position + count > len(self.child_items):
            return False

        for row in range(count):
            self.child_items.pop(position)

        return True

    def remove_columns(self, position: int, columns: int) -> bool:
        if position < 0 or position + columns > len(self.item_data):
            return False

        for column in range(columns):
            self.item_data.pop(position)

        for child in self.child_items:
            child.remove_columns(position, columns)

        return True

    def set_data(self, column: int, value, role):
        if column < 0 or column >= len(self.item_data):
            return False

        self.item_data[column] = value
        return True

    def __repr__(self) -> str:
        result = f"<{type(self).__name__} at 0x{id(self):x}"
        for d in self.item_data:
            result += f' "{d}"' if d else " <None>"
        result += f", {len(self.child_items)} children>"
        return result


class DataExplorerItem(TreeItem):
    COLUMNS = ["type", "time", "dispersion", "amplitude", "voltage", "beta", "masked"]
    MASK_COLUMN = COLUMNS.index("masked")
    SCAN_TYPE_TO_HEADER_STRING = {ScanType.DISPERSION: "Dispersion Scan",
                                  ScanType.TDS: "Voltage Scan",
                                  ScanType.BETA: "Beta Scan"}

    def __init__(self, name, timestamp=None, dispersion=None, amplitude=None, voltage=None, beta=None, masked=MaskState.NotMasked, parent: 'DataExplorerItem' = None):
        self.name = name
        self.timestamp = timestamp
        self.dispersion = dispersion
        self.amplitude = amplitude
        self.voltage = voltage
        self.beta = beta
        self._masked = masked

        self.parent_item = parent
        self.child_items = []

    @property
    def masked(self) -> MaskState:
        return self._masked

    @masked.setter
    def masked(self, value: MaskState):
        if not isinstance(value, MaskState):
            raise TypeError(f"value for masked is not of type MaskState: {value}")
        self._masked = value

    @property
    def time(self) -> str:
        """Returns prettyfied version of the timestamp"""
        try:
            dt = datetime.datetime.fromtimestamp(self.timestamp)
        except TypeError:
            return ""
        else:
            return dt.strftime("%H:%M:%S")

    @property
    def item_data(self):
        # XXX: Is there some way to suppress Masked being shown without just a magic empty string here?
        # XXX: Actually everything should be maskable, would be much cleaner
        voltage = self.voltage
        if self.voltage is not None:
            voltage *= 1e-6
            voltage = f"{voltage:.2g}"

        ampl = ""
        if self.amplitude is not None:
            ampl = f"{self.amplitude:.2g}"

        masked_string = ""
        return [self.name, self.time, self.dispersion, ampl, voltage, self.beta, masked_string]

    def data(self, column: int, role: int = None):
        if role == Qt.DisplayRole:
            return self.item_data[column]
        elif role == Qt.CheckStateRole and column == self.MASK_COLUMN:
            if self.masked is MaskState.Masked:
                return Qt.Checked
            elif self.masked is MaskState.NotMasked:
                return Qt.Unchecked
            elif self.masked is MaskState.PartiallyMasked:
                return Qt.PartiallyChecked
            else:
                raise ValueError("What?!")


        return self.item_data[column]

    def set_data(self, column: int, value, role) -> None:
        if column < 0 or column >= len(self.item_data):
            return False

        if column == self.MASK_COLUMN and role == Qt.CheckStateRole:
            if value == Qt.Checked:
                self.masked = MaskState.Masked
            elif value == Qt.Unchecked:
                self.masked = MaskState.NotMasked

            for irow, child in enumerate(self.child_items):
                child.masked = self.masked
        else:
            self.item_data[column] = value
        return True

    def is_column_checkable(self, column: int) -> bool:
        return column == self.MASK_COLUMN

    def flags(self, column):
        if self.is_column_checkable(column):
            return Qt.ItemIsUserCheckable
        else:
            return Qt.NoItemFlags


class ImageLeaf(DataExplorerItem):
    def __init__(self, timestamp, dbim, parent=None):
        super().__init__(name="Image", timestamp=timestamp, parent=parent)
        self.masked = MaskState.NotMasked
        self.dbim = dbim


class ScanNode(DataExplorerItem):
    SCAN_TYPE_TO_HEADER_STRING = {ScanType.DISPERSION: "Dispersion Scan",
                                  ScanType.TDS: "Voltage Scan",
                                  ScanType.BETA: "Beta Scan"}
    def __init__(self, scan_type: ScanType, parent=None):
        super().__init__(name=self.SCAN_TYPE_TO_HEADER_STRING[scan_type], parent=parent)
        self.scan_type = scan_type

    @property
    def scan_type(self) -> ScanType:
        return self._scan_type

    @scan_type.setter
    def scan_type(self, value: ScanType):
        self._scan_type = value
        self.item_data[0] = self.SCAN_TYPE_TO_HEADER_STRING[value]


class DataExplorerModel(QAbstractItemModel):
    HEADER_LABELS = ["Type", "Time", "D / m", "Amplitude Setpoint / %", "V / MV", "β / m", "Mask"]

    keep_timestamps_signal = pyqtSignal(set)
    skip_timestamps_signal = pyqtSignal(set)

    def __init__(self, measurement: MeasurementDataFrames, parent=None):
        super().__init__(parent)
        # self.root_item = DataExplorerItem(DataExplorerItem.HEADER_LABELS.copy())
        self.root_item = TreeItem(self.HEADER_LABELS.copy())

        self.setup_model_data(measurement)

        pipeline = make_image_processing_pipeline(bg=0.0)
        self.image_titles = ["Raw Image"] + [x.short for x in pipeline]

    def _emit_keep_timestamps_signal(self, timestamps: set) -> None:
        if not timestamps:
            return
        LOG.debug(f"Emitted keep_timestamps_signal: {timestamps}")
        self.keep_timestamps_signal.emit(timestamps)

    def _emit_skip_timestamps_signal(self, timestamps: set) -> None:
        if not timestamps:
            return
        LOG.debug(f"Emitted skip_timestamps_signal: {timestamps}")
        self.skip_timestamps_signal.emit(timestamps)

    def handle_skip_timestamps(self, timestamps: set):
        indices = [self.index_from_timestamp(t) for t in timestamps]
        indices.sort(key=lambda x: x.row())
        for index in sorted(indices):
            item = self.get_item(index)
            item.masked = MaskState.Masked

        self.dataChanged.emit(indices[0], indices[-1], [Qt.DisplayRole, Qt.CheckStateRole])

    def update_voltages(self, avmapping):
        def _update_item_and_children(item):
            try:
                amplitude = item.amplitude
            except AttributeError:
                pass
            else:
                if item.voltage is not None:
                    item.voltage = avmapping.get_voltage(amplitude)
            for child in item.child_items:
                _update_item_and_children(child)
        _update_item_and_children(self.root_item)

    def _item_from_timestamp(self, timestamp):
        def _get_items_from_timestamp(item, timestamp):
            result = []
            for child in item.child_items:
                try:
                    ts = child.timestamp
                except AttributeError:
                    pass
                else:
                    if ts == timestamp:
                        result.append(child)
                result.extend(_get_items_from_timestamp(child, timestamp))
            return result
        items = _get_items_from_timestamp(self.root_item, timestamp)
        # In principle could be multiple items with the same timestamp but we do accept this
        assert len(items) == 1
        return items[0]

    def f(self, item):
        for child in item.child_items:
            self.f(child)

    def index_from_timestamp(self, timestamp):
        item = self._item_from_timestamp(timestamp)
        # return self._index_from_item(item)
        parent = item.parent()
        irow = parent.child_items.index(item)
        return self.createIndex(irow, 0, item)

    # Method to find an item by name and return its QModelIndex
    def findItem(self, name, parent=QModelIndex()):
        if not parent.isValid():
            parentItem = self.root
        else:
            parentItem = parent.internalPointer()

        for row, child in enumerate(parentItem.children):
            if child.name == name:
                return self.createIndex(row, 0, child)
            else:
                foundIndex = self.findItem(name, self.createIndex(row, 0, child))
                if foundIndex.isValid():
                    return foundIndex

        return QModelIndex()

    def _index_from_item(self, item):
        def get_index(item):
            parent = item.parent()
            if parent is None:
                parent = QModelIndex()
            else:
                irow = parent.child_items.index(item)
                return self.createIndex(irow, 0, get_index(parent))
        return get_index(item)

    def get_decomposed_measurement(self) -> DecomposedMeasurement:
        # Now for each scan index we get all the Setpoint instances.
        decomposed_dscan = self._get_decomposed_scan_from_scan_type(ScanType.DISPERSION)
        decomposed_tscan = self._get_decomposed_scan_from_scan_type(ScanType.TDS)
        decomposed_bscan = self._get_decomposed_scan_from_scan_type(ScanType.BETA)

        return DecomposedMeasurement(decomposed_dscan, decomposed_tscan, decomposed_bscan)

    def _get_decomposed_scan_from_scan_type(self, scan_type: ScanType) -> DecomposedMeasurement:
        # For the given request scan_type, we get all of its corresponding setpoints and create Setpoint

        scan_index = self._get_scan_node_index(scan_type)
        scan_node = scan_index.internalPointer()
        decomposed_setpoints = []
        for irow, child in enumerate(scan_node.child_items):

            if child.name == "Setpoint":
                setpoint_index = self.index(irow, column=0, parent=scan_index)
                decomp_sp = self.get_decomposed_setpoint_from_setpoint_node(setpoint_index)
                decomposed_setpoints.append(decomp_sp)
        return DecomposedScan(scan_type, decomposed_setpoints)

    def get_decomposed_setpoint_from_setpoint_node(self, setpoint_node_index: QModelIndex) -> DecomposedSetpoint:
        """Given an index pointing to a Setpoint node, get the
        DecomposedSetpoing instance for all its daughter images rows."""
        # Local reference
        setpoint_node = setpoint_node_index.internalPointer()
        # What we return.
        # Get all the indices for the Images rows, there should only be one per setpoint
        images_node_indices = self.get_child_indices_by_type_string(setpoint_node_index, "Images")
        # We assert that there is indeed only one.
        assert len(images_node_indices) == 1
        # We get the index of the Images row
        images_node_index = images_node_indices[0]
        # Get the images Item
        images_node = images_node_index.internalPointer()

        # Iterate over the children and get all the image rows, create
        # an index, then a persistent index, and add it to the DecomposedSetpoint instance
        images = []
        for irow, child in enumerate(images_node.child_items):
            if child.name == "Image":
                images.append(child.dbim)

                # index = QPersistentModelIndex(self.index(irow, column=0, parent=images_node_index))
                # decomposed_setpoint.add_image(index)

        decomposed_setpoint = DecomposedSetpoint(images=images,
                                                 amplitude=setpoint_node.amplitude,
                                                 dispersion=setpoint_node.dispersion,
                                                 beta=setpoint_node.beta)

        return decomposed_setpoint

    def get_child_indices_by_type_string(self, index: QModelIndex, type_tag: str):
        row = self.get_item(index)
        result = []
        for irow, childrow in enumerate(row.child_items):
            this_row_type_tag = childrow.data(0)
            if this_row_type_tag == type_tag:
                result.append(self.index(irow, 0, index))
        return result

    def setup_model_data(self, measurement: MeasurementDataFrames) -> None:
        self._setup_scan_data(ScanType.DISPERSION, measurement.dscan)
        self._setup_scan_data(ScanType.TDS, measurement.tscan)
        self._setup_scan_data(ScanType.BETA, measurement.bscan)

    def _get_scan_node_index(self, scan_type: ScanType) -> QModelIndex:
        # From the top level get the ScanNode instance with the type
        # that matches this one.
        for irow, item in enumerate(self.root_item.child_items):
            try:
                item_scan_type = item.scan_type
            except AttributeError:
                pass
            else:
                if item_scan_type is scan_type:
                    return self.index(irow, column=0, parent=QModelIndex())
        raise ValueError(f"Unable to find top level ScanNode instance with scan_type: {scan_type}")

        #     # index = self.HEADER_LABELS.index("Type")
        #     # irow = next(i for (i, x) in enumerate(self.root_item.child_items) if x.data(index) == header_string)
        # return self.index(irow, column=0, parent=QModelIndex())

    def _make_scan_node(self, scan_type: ScanType) -> ScanNode:
        """Make a node for a Scan, i.e. dispersion, beta or tds scan"""
        scan_node = ScanNode(scan_type, parent=self.root_item)
        self.root_item.child_items.append(scan_node)
        return scan_node

    def _setup_scan_data(self, scan_name: str, scan):
        scan_node = self._make_scan_node(scan_name)

        for i, sp in enumerate(scan.setpointdfs):
            setpoint_node = DataExplorerItem(name="Setpoint",
                                             timestamp=None,
                                             dispersion=sp.dispersion,
                                             amplitude=sp.amplitude_sp,
                                             voltage=sp.voltage,
                                             beta=sp.beta,
                                             masked=MaskState.NotMasked,
                                             parent=scan_node)

            self._setup_scan_setpoint(sp, parent=setpoint_node)
            scan_node.child_items.append(setpoint_node)

    def _setup_scan_setpoint(self, sp, parent):
        pipeline = make_image_processing_pipeline(bg=sp.bg)

        # Add Analysis Breakdown Here
        # image_ana = DataExplorerItem(data=len(self.HEADER_LABELS) * [None], parent=parent)
        # image_ana.item_data[0] = "Image Analysis Breakdown"
        # parent.appendRow(image_ana)
        # image_ana_leaf = DataExplorerItem(data=["Image Analysis Breakdown"], parent=parent)
        # parent.child_items.append(image_ana_leaf)

        # Add Images
        images_node = DataExplorerItem(name="Images", parent=parent, masked=MaskState.NotMasked)
        for tim in sp.get_tagged_images():
            try:
                timestamp = tim.metadata.kvps["timestamp"]
            except KeyError:
                LOG.warning("Missing timestamp in DF, using arbitrary one isntead.")
                timestamp = time.time() + np.random.rand()

            datetime.datetime.fromtimestamp(timestamp)

            dbim = DecomposedBeamImage(pipeline, tim)
            image_leaf = ImageLeaf(timestamp=timestamp, dbim=dbim, parent=images_node)
            images_node.child_items.append(image_leaf)


        parent.child_items.append(images_node)

    def _repr_recursion(self, item: DataExplorerItem, indent: int = 0) -> str:
        result = " " * indent + repr(item) + "\n"
        for child in item.child_items:
            result += self._repr_recursion(child, indent + 2)
        return result

    def __repr__(self) -> str:
        return self._repr_recursion(self.root_item)

    def columnCount(self, parent: QModelIndex = None) -> int:
        return self.root_item.column_count()

    def data(self, index: QModelIndex, role: int = None):
        if not index.isValid():
            return None
        if role != Qt.DisplayRole and role != Qt.CheckStateRole:
            return None

        item: DataExplorerItem = self.get_item(index)

        if not item.is_column_checkable(index.column()) and role == Qt.CheckStateRole:
            return None

        return item.data(index.column(), role=role)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.NoItemFlags
        # I don't want any part of the table to be editable so I do not include ItemIsEditable
        item = self.get_item(index)

        return QAbstractItemModel.flags(self, index) | Qt.ItemIsEnabled | item.flags(index.column())

    def get_item(self, index: QModelIndex = QModelIndex()) -> DataExplorerItem:
        if index.isValid():
            item: DataExplorerItem = index.internalPointer()
            if item:
                return item

        return self.root_item

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.root_item.data(section)

        return None

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if parent.isValid() and parent.column() != 0:
            return QModelIndex()

        parent_item: DataExplorerItem = self.get_item(parent)
        if not parent_item:
            return QModelIndex()

        child_item: DataExplorerItem = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def insertColumns(self, position: int, columns: int,
                      parent: QModelIndex = QModelIndex()) -> bool:
        self.beginInsertColumns(parent, position, position + columns - 1)
        success: bool = self.root_item.insert_columns(position, columns)
        self.endInsertColumns()

        return success

    def insertRows(self, position: int, rows: int,
                   parent: QModelIndex = QModelIndex()) -> bool:
        parent_item: DataExplorerItem = self.get_item(parent)
        if not parent_item:
            return False

        self.beginInsertRows(parent, position, position + rows - 1)
        column_count = self.root_item.column_count()
        success: bool = parent_item.insert_children(position, rows, column_count)
        self.endInsertRows()

        return success

    def parent(self, index: QModelIndex = QModelIndex()) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()

        child_item: DataExplorerItem = self.get_item(index)
        if child_item:
            parent_item: DataExplorerItem = child_item.parent()
        else:
            parent_item = None

        if parent_item == self.root_item or not parent_item:
            return QModelIndex()

        return self.createIndex(parent_item.child_number(), 0, parent_item)

    def removeColumns(self, position: int, columns: int,
                      parent: QModelIndex = QModelIndex()) -> bool:
        self.beginRemoveColumns(parent, position, position + columns - 1)
        success: bool = self.root_item.remove_columns(position, columns)
        self.endRemoveColumns()

        if self.root_item.column_count() == 0:
            self.removeRows(0, self.rowCount())

        return success

    def removeRows(self, position: int, rows: int,
                   parent: QModelIndex = QModelIndex()) -> bool:
        parent_item: DataExplorerItem = self.get_item(parent)
        if not parent_item:
            return False

        self.beginRemoveRows(parent, position, position + rows - 1)
        success: bool = parent_item.remove_children(position, rows)
        self.endRemoveRows()

        return success

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid() and parent.column() > 0:
            return 0

        parent_item: DataExplorerItem = self.get_item(parent)
        if not parent_item:
            return 0
        return parent_item.child_count()

    def setData(self, index: QModelIndex, value, role: int) -> bool:
        if role != Qt.CheckStateRole:
            return False

        item: DataExplorerItem = self.get_item(index)
        result: bool = item.set_data(index.column(), value, role)

        if value == Qt.Checked:
            self._emit_skip_timestamps_signal({item.timestamp})
        elif value == Qt.Unchecked:
            self._emit_keep_timestamps_signal({item.timestamp})

        if result:
            self.dataChanged.emit(index, index, [Qt.DisplayRole, Qt.CheckStateRole])
            self.update_child_checkboxes(index)
            self.update_parent_checkboxes(index)
            # self.dataChanged.emit(index, index, [])

        return result

    def update_child_checkboxes(self, index0):
        # we assert all items are always checkable in our model,
        # because we always have one checkbox at the end for masking.
        item = self.get_item(index0)
        try:
            masked = item.masked
        except AttributeError:
            return
        index1 = index0

        timestamps_to_be_skipped = set()
        timestamps_to_be_kept = set()

        for irow, child in enumerate(item.child_items):
            child.masked = masked
            child_index = self.createIndex(irow, 0, child)
            self.update_child_checkboxes(child_index)
            index1 = child_index

            try:
                ts = child.timestamp
            except AttributeError:
                pass
            else:
                if masked is MaskState.Masked:
                    timestamps_to_be_skipped.add(ts)
                else:
                    timestamps_to_be_kept.add(ts)

        self._emit_skip_timestamps_signal(timestamps_to_be_skipped)
        self._emit_keep_timestamps_signal(timestamps_to_be_kept)
        self.dataChanged.emit(index0, index1, [Qt.CheckStateRole])

    def update_parent_checkboxes(self, index0):
        parent_index = self.parent(index0)
        parent_item = parent_index.internalPointer()

        if parent_item is None:
            return

        masked_siblings = [sibling.masked is MaskState.Masked for sibling in parent_item.child_items]
        partially_masked = [sibling.masked is MaskState.PartiallyMasked for sibling in parent_item.child_items]

        any_masked = any(masked_siblings)
        all_masked = all(masked_siblings)
        any_partially_masked = any(partially_masked)

        try:
            masked = parent_item.masked
        except AttributeError:
            return

        if all_masked:
            masked = MaskState.Masked
        elif any_masked or any_partially_masked:
            masked = MaskState.PartiallyMasked
        else:
            masked = MaskState.NotMasked

        parent_item.masked = masked
        self.update_parent_checkboxes(parent_index) # Recurse up tree.
        self.dataChanged.emit(QModelIndex(), index0, [Qt.CheckStateRole])

    def setHeaderData(self, section: int, orientation: Qt.Orientation, value,
                      role: int = None) -> bool:
        if orientation != Qt.Horizontal:
            return False

        result: bool = self.root_item.set_data(section, value)

        if result:
            self.headerDataChanged.emit(orientation, section, section)

        return result


class ImageProcessingStepImpactWidget(QMainWindow):
    STAGE_MAP = ["Raw Image",
                 "Background Subtraction",
                 "Uniform Filtered: 100",
                 "Uniform Filtered: 3",
                 "Remove isolated pixels",
                 "Outlier images filtered"]
    selected_timestamp = pyqtSignal(float)

    SCAN_TYPE_COMBO_LABELS = ["Dispersion", "TDS", "Beta"]
    SCAN_LABEL_TO_SCAN_TYPE = {"Dispersion": ScanType.DISPERSION,
                               "TDS": ScanType.TDS,
                               "Beta": ScanType.BETA}

    def __init__(self, decomposed_measurement: DecomposedMeasurement, parent=None):
        super().__init__()
        self.decomposed_measurement = decomposed_measurement

        self._xdict = dict(enumerate(self.STAGE_MAP))

        self.init_ui()

        self.plot_items_to_timestamps = {}
        self._selected_plot_item = None


    def init_ui(self):
        # Central widget and layout setup
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Plotting widget
        plot_widget = pg.GraphicsLayoutWidget()
        # Add GraphicsLayoutWidget for plotting
        plot_widget = pg.GraphicsLayoutWidget()
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([self._xdict.items()])

        self.plot_item = plot_widget.addPlot(title="Plot", axisItems={'bottom': stringaxis})
        self.plot_item.setLabel("left", "<i>&sigma;<sub>M</sub></i>", units="px")
        self.plot_item.setLabel("bottom", "Image Processing Stage")
        main_layout.addWidget(plot_widget)

        # Controls layout
        controls_layout = QHBoxLayout()
        main_layout.addLayout(controls_layout)

        # Scan Type dropdown
        scan_type_label = QLabel("Scan Type")
        self.scan_type_combo = QComboBox()
        self.scan_type_combo.addItems(self.SCAN_TYPE_COMBO_LABELS)
        controls_layout.addWidget(scan_type_label)
        controls_layout.addWidget(self.scan_type_combo)

        # Dispersion, Voltage, Beta dropdowns
        self.dispersion_label = QLabel("Dispersion / m")
        self.dispersion_combo = QComboBox()
        # self.dispersion_combo.addItems(dispersions)
        self.voltage_label = QLabel("Voltage / MV")
        self.voltage_combo = QComboBox()
        # self.voltage_combo.addItems(voltages)
        self.beta_label = QLabel("β / m")
        self.beta_combo = QComboBox()
        # self.beta_combo.addItems(betas)

        self.set_dropdowns(ScanType.DISPERSION)

        # Adding widgets to layout
        controls_layout.addWidget(self.dispersion_label)
        controls_layout.addWidget(self.dispersion_combo)
        controls_layout.addWidget(self.voltage_label)
        controls_layout.addWidget(self.voltage_combo)
        controls_layout.addWidget(self.beta_label)
        controls_layout.addWidget(self.beta_combo)

        # Navigation buttons
        self.go_button = QPushButton("Go")
        self.previous_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        controls_layout.addWidget(self.go_button)
        controls_layout.addWidget(self.previous_button)
        controls_layout.addWidget(self.next_button)

        # Button signals
        self.go_button.clicked.connect(self.handle_go)
        self.next_button.clicked.connect(self.handle_next)
        self.previous_button.clicked.connect(self.handle_previous)

        # Initial state setup
        self.scan_type_combo.currentIndexChanged.connect(self.set_dropdowns)
        self.set_dropdowns()

    def set_dropdowns(self, _=None):
        self.dispersion_combo.setEnabled(True)
        self.voltage_combo.setEnabled(True)
        self.beta_combo.setEnabled(True)

        scan_type = self.get_active_scan_type()

        if scan_type is ScanType.DISPERSION:
            dispersions = self.decomposed_measurement.dscan_dispersions()
            voltages = [self.decomposed_measurement.tscan.setpoints[0].voltage]
            betas = [self.decomposed_measurement.dscan.setpoints[0].beta]
            self.beta_combo.setEnabled(False)
            self.voltage_combo.setEnabled(False)
        elif scan_type is ScanType.TDS:
            dispersions = [self.decomposed_measurement.tscan.setpoints[0].dispersion]
            voltages = self.decomposed_measurement.tscan_voltages()
            betas = [self.decomposed_measurement.dscan.setpoints[0].beta]
            self.dispersion_combo.setEnabled(False)
            self.beta_combo.setEnabled(False)
        elif scan_type is ScanType.BETA:
            dispersions = [self.decomposed_measurement.dscan.setpoints[0].dispersion]
            voltages = [self.decomposed_measurement.tscan.setpoints[0].voltage]
            betas = self.decomposed_measurement.bscan_betas()
            self.dispersion_combo.setEnabled(False)
            self.voltage_combo.setEnabled(False)

        self.dispersion_combo.clear()
        self.dispersion_combo.addItems([str(x) for x in dispersions])
        self.voltage_combo.clear()
        self.voltage_combo.addItems([str(x) for x in voltages])
        self.beta_combo.clear()
        self.beta_combo.addItems([str(x) for x in betas])

    def get_active_scan_type(self) -> ScanType:
        scan_string = self.scan_type_combo.currentText()
        return self.SCAN_LABEL_TO_SCAN_TYPE[scan_string]

    def handle_go(self):
        self.plot_items_to_timestamps = {}
        setpoint = self.get_setpoint_from_gui_state()
        widths = setpoint.get_all_widths_at_all_stages()
        self.plot_widths(widths, setpoint.timestamps())


    def handle_next(self):
        self.navigate_dropdowns(next=True)

    def handle_previous(self):
        self.navigate_dropdowns(next=False)

    def navigate_dropdowns(self, next):
        # Determine the active dropdown and its current index
        active_dropdown = self.get_active_dropdown()
        current_index = active_dropdown.currentIndex()

        if next:
            if current_index < active_dropdown.count() - 1:
                # Move to next item if not at the end
                active_dropdown.setCurrentIndex(current_index + 1)
            else:
                # Move to next Scan Type and reset to the first item of the new dropdown
                self.cycle_scan_type(next=True)
                self.reset_active_dropdown(start_from_top=True)
        else:
            if current_index > 0:
                # Move to previous item if not at the start
                active_dropdown.setCurrentIndex(current_index - 1)
            else:
                # Move to previous Scan Type and reset to the last item of the new dropdown
                self.cycle_scan_type(next=False)
                self.reset_active_dropdown(start_from_top=False)

    def reset_active_dropdown(self, start_from_top):
        # Reset the current index of the active dropdown based on direction
        active_dropdown = self.get_active_dropdown()
        if start_from_top:
            # Set to the first item
            active_dropdown.setCurrentIndex(0)
        else:
            # Set to the last item
            active_dropdown.setCurrentIndex(active_dropdown.count() - 1)

    def _get_dispersion_from_gui(self) -> float:
        return float(self.dispersion_combo.currentText())

    def _get_voltage_from_gui(self) -> float:
        return float(self.voltage_combo.currentText()) * 1e6 # Convert to MV.

    def _get_beta_from_gui(self) -> float:
        return float(self.beta_combo.currentText())

    def get_setpoint_from_gui_state(self) -> DecomposedSetpoint:
        scan_type = self.get_active_scan_type()
        if scan_type is ScanType.DISPERSION:
            scan = self.decomposed_measurement.dscan
            dispersion = self._get_dispersion_from_gui()
            index = np.argmin(abs(scan.dispersions() - dispersion))
        elif scan_type is ScanType.TDS:
            scan = self.decomposed_measurement.tscan
            voltage = self._get_voltage_from_gui()
            index = np.argmin(abs(scan.voltages() - voltage))
        elif scan_type is ScanType.BETA:
            scan = self.decomposed_measurement.bscan
            beta = self._get_beta_from_gui()
            index = np.argmin(abs(scan.betas() - beta))

        return scan.setpoints[index]

    def get_active_dropdown(self):
        # Return the currently active dropdown based on Scan Type
        selection = self.scan_type_combo.currentText()
        if selection == "Dispersion":
            return self.dispersion_combo
        elif selection == "TDS":
            return self.voltage_combo  # Assuming TDS affects Voltage
        elif selection == "Beta":
            return self.beta_combo

    def cycle_scan_type(self, next):
        # Cycle the Scan Type selection
        current_index = self.scan_type_combo.currentIndex()
        if next:
            new_index = (current_index + 1) % self.scan_type_combo.count()
        else:
            new_index = (current_index - 1) % self.scan_type_combo.count()
        self.scan_type_combo.setCurrentIndex(new_index)

    def plot_widths(self, widths, timestamps):
        for setpoint_stage_widths, timestamp in zip(widths, timestamps):
            plot_data_item = self.plot_item.plot(list(self._xdict), setpoint_stage_widths)
            plot_data_item.setCurveClickable(True)
            plot_data_item.sigClicked.connect(self.handle_click)
            self.set_default_pen(plot_data_item)
            self.plot_items_to_timestamps[plot_data_item] = timestamp

    @staticmethod
    def set_default_pen(line_item):
        default_pen = pg.mkPen(color=pg.mkColor("white"), width=5)
        line_item.setPen(default_pen)

    @staticmethod
    def set_selected_line_pen(line_item):
        highlighter_pen = pg.mkPen(color=pg.mkColor((223, 125, 124)), width=5)
        line_item.setPen(highlighter_pen)

    def handle_click(self, clicked_plot_data_item):
        timestamp = self.plot_items_to_timestamps[clicked_plot_data_item]
        self.selected_timestamp.emit(timestamp)
        # If clicking on what we have already selected, then do nothing.
        previously_selected_plot_item = self._selected_plot_item

        try:
            self.set_default_pen(previously_selected_plot_item)
        except AttributeError:
            pass

        self._selected_plot_item = clicked_plot_data_item
        self.set_selected_line_pen(self._selected_plot_item)


class PlotSelecterWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # Layouts
        main_layout = QVBoxLayout(self)
        combo_layout = QHBoxLayout()
        
        # Combo Box with Labels and Button
        self.plot_label = QLabel("Plot:")
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Option 1", "Option 2", "Option 3"])  # Add your options here
        
        # Show Button
        self.show_button = QPushButton("Show")
        self.show_button.clicked.connect(self.on_show_clicked)
        
        # Adding widgets to the combo layout
        combo_layout.addWidget(self.plot_label)
        combo_layout.addWidget(self.combo_box)
        combo_layout.addWidget(self.show_button)
        
        # Adding combo layout to the main layout
        main_layout.addLayout(combo_layout)
        
    def on_show_clicked(self):
        # Placeholder for action based on combo box selection
        selected_option = self.combo_box.currentText()
        print(f"Selected: {selected_option}")  # Implement your action here

    
