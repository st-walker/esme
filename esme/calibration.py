import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.constants import c, e
from scipy.optimize import curve_fit

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ocelot.cpbd.magnetic_lattice import MagneticLattice

from esme.lattice import injector_cell_from_snapshot
from esme.maths import line

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH
TDS_LENGTH = 0.7  # metres
TDS_PERIOD = 1 / TDS_FREQUENCY


def sqrt(x, a0, a1) -> Any:
    return a0 + a1 * np.sqrt(x)


class TDSCalibrator:
    def __init__(
        self,
        percentages: Sequence[float],
        tds_slopes: Sequence[float],
        dispersion_setpoint: float,
        tds_slope_units: Optional[str] = None,
    ):
        self.percentages = np.array(percentages)
        self.tds_slopes = np.array(tds_slopes)
        self.dispersion_setpoint = dispersion_setpoint
        if tds_slope_units == "um/ps":
            self.tds_slopes = self.tds_slopes * 1e6
        elif tds_slope_units is not None:
            raise TypeError(f"Wrong tds_slope_units: {tds_slope_units}")

    def fit(self):
        popt, pcov = curve_fit(line, self.percentages, self.tds_slopes)
        return popt, pcov

    def get_tds_slope(self, percentage: float) -> float:
        popt, _ = self.fit()
        return line(percentage, *popt)

    def get_voltage(
        self, percentage: Union[float, Sequence[float]], snapshot: pd.Series
    ) -> Union[float, Sequence[float]]:
        tds_slope = self.get_tds_slope(percentage)
        return get_tds_voltage(tds_slope, snapshot)

    def __repr__(self) -> str:
        cname = type(self).__name__
        dx0 = self.dispersion_setpoint
        return f"<{cname}: {dx0=}, %={repr(self.percentages)}, grds={self.tds_slopes}>"


class TrivialTDSCalibrator:
    def __init__(self, percentages: Sequence[float], voltages: Sequence[float]):
        self.percentages = np.array(percentages)
        self.voltages = np.array(voltages)

    def get_voltage(self, percentage) -> float:
        return dict(zip(self.percentages, self.voltages))[percentage]


class TDSVoltageCalibration:
    """full TDS calibration mapping amplitudes to voltages or slopes,
    requires no additional calculation besides interpolating between
    the data points (i.e. no needs no snapshots)

    """

    def __init__(self, amplitudes: Sequence[float], voltages):
        self.amplitudes = amplitudes
        self.voltages = voltages

    def fit_voltages(self):
        popt, pcov = curve_fit(line, self.percentages, self.voltages)
        return popt, pcov

    def get_voltage(self, amplitude: float) -> float:
        popt, _ = self.fit_voltages()
        return line(amplitude, *popt)


def lat_from_tds_to_screen(snapshot: pd.Series):
    cell = injector_cell_from_snapshot(snapshot)
    screen_marker = next(ele for ele in cell if ele.id == "OTRC.64.I1D")
    tds_marker = next(ele for ele in cell if ele.id == "TDS2")

    lat = MagneticLattice(cell, start=tds_marker, stop=screen_marker)

    return lat


def r34_from_tds_to_screen(snapshot: pd.Series) -> float:
    lat = lat_from_tds_to_screen(snapshot)
    energy = snapshot[I1D_ENERGY_ADDRESS].mean()

    # Convert energy to GeV from MeV
    _, rmat, _ = lat.transfer_maps(energy * 1e-3)
    r34 = rmat[2, 3]
    return r34


def get_tds_voltage(
    gradient_m_per_s: Union[float, Sequence[float]], snapshot: pd.Series
) -> Union[float, Sequence[float]]:
    r34 = r34_from_tds_to_screen(snapshot)
    energy = snapshot[I1D_ENERGY_ADDRESS]
    angular_frequency = TDS_FREQUENCY * 2 * np.pi  # to rad/s
    energy_joules = energy * e * 1e6  # Convert to joules.
    voltage = (energy_joules / (e * angular_frequency * r34)) * gradient_m_per_s
    return voltage


def r34s_from_scan(scan):
    result = []
    for measurement in scan:
        # Pick a non-bg image.
        im = measurement.images[0]
        result.append(r34_from_tds_to_screen(im.metadata))
    return np.array(result)
