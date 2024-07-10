from dataclasses import dataclass
from typing import SimpleNamespace

import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QModelIndex, Qt
from scipy.constants import c

from esme.plot import _format_df_for_printing


class InterruptedMeasurement(RuntimeError):
    pass


@dataclass
class MeasuredBeamParameters:
    sigma_z: tuple[float, float]
    resolution_z: tuple[float, float]
    sigma_x0: tuple[float, float]
    sigma_xi: tuple[float, float]

    @property
    def resolution_t(self):
        return (self.resolution_z[0] / c, self.resolution_z[1] / c)

    @property
    def sigma_t(self):
        return (self.sigma_z[0] / c, self.sigma_z[1] / c)


def format_value_with_uncertainty(value, uncertainty):
    """Format the value with its uncertainty."""
    return f"{value} ± {uncertainty}"


class BeamCurrentTableModel(QAbstractTableModel):
    def __init__(
        self,
        gaussian_params: MeasuredBeamParameters | None = None,
        rms_params: MeasuredBeamParameters | None = None,
    ):
        super().__init__()

        self.gaussian_params = gaussian_params or MeasuredBeamParameters()
        self.rms_params = rms_params or MeasuredBeamParameters()

        self.headers = ["Gaussian", "RMS", "Units"]
        self.param_order = [
            "sigma_z",
            "sigma_t",
            "resolution_z",
            "resolution_t",
            "sigma_x0",
            "sigma_xi",
        ]
        self.units = {
            "sigma_z": "mm",
            "sigma_t": "s",
            "resolution_z": "mm",
            "resolution_t": "s",
            "sigma_x0": "mm",
            "sigma_xi": "mm",
        }
        self.html_headers = [
            "<i>σ<sub>z</sub></i>",
            "<i>σ<sub>t</sub></i>",
            "<i>R<sub>z</sub></i>",
            "<i>R<sub>t</sub></i>",
            "<i>σ<sub>x,0</sub></i>",
            "<i>σ<sub>x,i</sub></i>",
        ]

    def rowCount(self, parent=None):
        return len(self.param_order)

    def columnCount(self, parent=None):
        return 3

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            param_name = self.param_order[row]
            gaussian_value = getattr(self.gaussian_params, param_name)
            rms_value = getattr(self.rms_params, param_name)

            if col == 0:
                return format_value_with_uncertainty(
                    gaussian_value[0], gaussian_value[1]
                )
            elif col == 1:
                return format_value_with_uncertainty(rms_value[0], rms_value[1])
            elif col == 2:
                return self.units[param_name]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole) -> str | None:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            elif orientation == Qt.Vertical:
                return self.html_headers[section]
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table model to a pandas DataFrame."""
        data = {
            "Parameter": [],
            "Gaussian Value": [],
            "Gaussian Uncertainty": [],
            "RMS Value": [],
            "RMS Uncertainty": [],
            "Units": [],
        }

        for row in range(self.rowCount()):
            param_name = self.param_order[row]
            gaussian_value = getattr(self.gaussian_params, param_name)
            rms_value = getattr(self.rms_params, param_name)

            data["Parameter"].append(param_name)
            data["Gaussian Value"].append(gaussian_value[0])
            data["Gaussian Uncertainty"].append(gaussian_value[1])
            data["RMS Value"].append(rms_value[0])
            data["RMS Uncertainty"].append(rms_value[1])
            data["Units"].append(self.units[param_name])

        return pd.DataFrame(data).set_index("Parameter")

    def to_logbook_printable_table(self):
        return _format_df_for_printing(
            self.to_dataframe(),
            [
                ("Gaussian Value", "Gaussian Uncertainty"),
                ("RMS Value", "RMS Uncertainty"),
            ],
            self.units,
        )


class CurrentProfileWorkerSignals(QObject):
    images_taken_signal = pyqtSignal(int)


class CurrentProfileWorker(QRunnable):
    def __init__(self, machine_interface, screen):
        super().__init__()

        self.machine = machine_interface
        self.screen = screen

        self.signals = CurrentProfileWorkerSignals()

        beam_images_per_streak = 10
        bg_images_per_gain = 5

    def run(self) -> None:
        try:
            self._measure_current_profile()
        except InterruptedMeasurement:
            pass

    def n_images(self) -> int:
        return int(self.beam_images_per_streak * 3 + self.bg_images_per_gain * 2)

    def _measure_current_profile(self) -> None:
        # First turn beam on and do auto gain
        # Then take images.

        # Then turn beam off and take background.

        # Then turn beam on and turn tds on do gain control.

        # Then flip tds phase and take images.

        # Then turn beam off and take background.

        pass


@dataclass
class CurrentProfileMeasurement:
    context = CalibrationContext
    bunch_charge: float = np.nan
    phase0: float = np.nan
    phase1: float = np.nan
    images_streak0: npt.NDArray
    images_streak1: npt.NDArray
    images_unstreaked: npt.ArrayLike
    background_streaked: npt.ArrayLike = 0.0
    background_unstreaked: npt.ArrayLike = 0.0
    tds_calibration: tuple[list[float], list[float]]
    tds_calibration_directory: Path

    def current0(self) -> tuple[np.ndarray, np.ndarray]:
        pass

    def current1(self) -> tuple[np.ndarray, np.ndarray]:
        pass


class CurrentProfilerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.ui = Ui_CurrentProfilerWindow()
        self.ui.setupUi(self)
        self.ui.results_table_view.setModel(BeamParametersTableModel())

        self.plots = SimpleNamespace()
        self._init_plots()

        self._measurement = None

    def _connect_buttons(self) -> None:
        self.ui.start_measurement_button.clicked.connect(self.measure_current_profile)
        self.ui.cancel_button.clicked.connect(self.cancel_current_profile_measurement)

    def measure_current_profile(self) -> None:
        pass

    def cancel_current_profile_measurement(self) -> None:
        pass

    def plot(self) -> None:
        pass

    def plot_current_profile(self, current_profile: np.ndarray) -> None:
        pass

    def clear_displays(self) -> None:
        self.plots.current.clear()
        self.plots.spot_size.clear()

    def _init_plots(self) -> None:
        self.plots.current = self.ui.current_graphics.addPlot(title="Current Profile")
        self.plots.spot_size = self.ui.current_graphics.addPlot(
            title="Streaked Spot Sizes"
        )

        self.plots.current.addLegend()
        self.plots.spot_size.addLegend()

        self.plots.current.setLabel("left", "<i>I</i>", units="A")
        self.plots.current.setLabel("bottom", "<i>t</i>", units="s")

        self.plots.spot_size.setLabel("bottom", "<i>S/|S|</i>")
        self.plots.spot_size.setLabel(
            "left", "Spot Size <i>σ<sub>x</sub></i>", units="m"
        )

    def save_result(self) -> None:
        pass
