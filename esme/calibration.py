import warnings
from typing import Any, Optional, Sequence, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.constants import c, e
from scipy.optimize import curve_fit
from functools import partial

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from ocelot.cpbd.magnetic_lattice import MagneticLattice

from esme.maths import line

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH
TDS_LENGTH = 0.7  # metres
TDS_PERIOD = 1 / TDS_FREQUENCY


def sqrt(x, a0, a1) -> Any:
    return a0 + a1 * np.sqrt(x)


# class TDSCalibrator:
#     def __init__(
#         self,
#         percentages: Sequence[float],
#         tds_slopes: Sequence[float],
#         dispersion_setpoint: float,
#         tds_slope_units: Optional[str] = None,
#     ):
#         self.percentages = np.array(percentages)
#         self.tds_slopes = np.array(tds_slopes)
#         self.dispersion_setpoint = dispersion_setpoint
#         if tds_slope_units == "um/ps":
#             self.tds_slopes = self.tds_slopes * 1e6
#         elif tds_slope_units is not None:
#             raise TypeError(f"Wrong tds_slope_units: {tds_slope_units}")

#     def get_tds_slope(self, percentage: float) -> float:
#         popt, _ = self.fit()
#         return line(percentage, *popt)

#     def get_voltage(
#         self, percentage: Union[float, Sequence[float]], snapshot: pd.Series
#     ) -> Union[float, Sequence[float]]:
#         tds_slope = self.get_tds_slope(percentage)
#         return get_tds_voltage(tds_slope, snapshot)

#     def __repr__(self) -> str:
#         cname = type(self).__name__
#         dx0 = self.dispersion_setpoint
#         return f"<{cname}: {dx0=}, %={repr(self.percentages)}, grds={self.tds_slopes}>"

# class CalibrationMapping:
#     def __init__(self, amplitudes, voltages):
#         self.amplitudes = amplitudes
#         self.voltages = voltages

#         popt, _ = curve_fit(line, amplitudes, voltages)
#         self.get_voltage = partial(line, a0=popt[0], a1=popt[1])
#         popt, _ = curve_fit(line, voltages, amplitudes)
#         self.get_amplitude = partial(line, a0=popt[0], a1=popt[1])

#     def get_voltages(self):
#         return np.array(self.voltages)

#     def get_amplitudes(self):
#         return np.array(self.amplitudes)

#     def get_voltage_fit_line(self):
#         amplitudes = np.linspace(0, max(max(self.amplitudes) * 1.1, 25))
#         fit_voltages = self.get_voltage(amplitudes)
#         return amplitudes, fit_voltages

#     def get_voltage_fit_parameters(self):
#         popt, _ = curve_fit(line, self.amplitudes, self.voltages)
#         return popt



class TDSCalibration:
    def get_voltage(self, amplitude):
        popt, _ = self.fit_to_voltage()
        return line(amplitude, *popt)

    def get_amplitude(self, voltage):
        popt, _ = self.fit_to_amplitude()
        return line(voltage, *popt)

    def fit_to_voltage(self):
        popt, pcov = curve_fit(line, self.get_amplitudes(), self.get_amplitudes())
        return popt, pcov

    def fit_to_amplitude(self):
        popt, pcov = curve_fit(line, self.get_voltages(), self.get_amplitudes())
        return popt, pcov
    

class StuartCalibration(TDSCalibration):
    def __init__(self, amplitudes, voltages):
        self.amplitudes = amplitudes
        self.voltages = voltages

    def get_voltages(self):
        return self.voltages

    def get_amplitudes(self):
        return self.amplitudes

class BolkoCalibrationSetPoint:
    def __init__(self, amplitude, slope, r34, energy, frequency):
        self.amplitude = amplitude
        self.slope = slope
        self.r34 = r34
        self.energy = energy
        self.frequency = frequency

    def __repr__(self):
        amp = self.amplitude
        slope = self.slope
        r34 = self.r34
        energy = self.energy
        freq = self.frequency
        return f"<BolkoCalibrationSetpoint: {amp=}, {slope=}, {r34=}, {energy=}, {freq=}>"

    def get_voltage(self):
        return calculate_voltage(slope=self.slope,
                                 r34=self.r34,
                                 energy=self.energy,
                                 frequency=self.frequency)

class BolkoCalibration(TDSCalibration):
    def __init__(self, bolko_setpoints):
        self.setpoints = bolko_setpoints

    def get_amplitudes(self):
        return np.array([setpoint.amplitude for setpoint in self.setpoints])

    def get_voltages(self):
        return np.array([setpoint.get_voltage() for setpoint in self.setpoints])


class IgorCalibration:
    def __init__(self, amplitudes, voltages):
        self.amplitudes = amplitudes
        self.voltages = voltages

    def get_amplitude(self, voltage):
        return dict(zip(self.voltages, self.amplitudes))[voltage]


class DiscreteCalibration(TDSCalibration):
    def __init__(self, amplitudes, voltages):
        self.amplitudes = amplitudes
        self.voltages = voltages

    def get_voltages(self):
        return np.array([sp.voltage for sp in self.setpoints])

    def get_amplitudes(self):
        return np.array([sp.amplitude for sp in self.setpoints])

    def get_voltage(self, amplitude):
        return dict(zip(self.amplitudes, self.voltages))[amplitude]

    def get_amplitude(self, voltage):
        return dict(zip(self.voltages, self.amplitudes))[voltage]



# class TDSCalibration:
#     def __init__(
#         self,
#             amplitudes,
#             phases,
#             centres_of_mass,
#         percentages: Sequence[float],
#         slopes: Sequence[float],
#         dispersion_setpoint: float,
#         slope_units: Optional[str] = None,
#     ):
#         self.percentages = np.array(percentages)
#         self.tds_slopes = np.array(tds_slopes)
#         self.dispersion_setpoint = dispersion_setpoint
#         if tds_slope_units == "um/ps":
#             self.tds_slopes = self.tds_slopes * 1e6
#         elif tds_slope_units is not None:
#             raise TypeError(f"Wrong tds_slope_units: {tds_slope_units}")

#     def fit(self):
#         popt, pcov = curve_fit(line, self.percentages, self.tds_slopes)
#         return popt, pcov

#     def get_tds_slope(self, percentage: float) -> float:
#         popt, _ = self.fit()
#         return line(percentage, *popt)

#     def get_voltage(
#         self, percentage: Union[float, Sequence[float]], snapshot: pd.Series
#     ) -> Union[float, Sequence[float]]:
#         tds_slope = self.get_tds_slope(percentage)
#         return get_tds_voltage(tds_slope, snapshot)

#     def __repr__(self) -> str:
#         cname = type(self).__name__
#         dx0 = self.dispersion_setpoint
#         return f"<{cname}: {dx0=}, %={repr(self.percentages)}, grds={self.tds_slopes}>"


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


def get_tds_voltage(
    gradient_m_per_s: Union[float, Sequence[float]], snapshot: pd.Series
) -> Union[float, Sequence[float]]:
    r34 = r34_from_tds_to_screen(snapshot)
    energy = snapshot[I1D_ENERGY_ADDRESS]
    angular_frequency = TDS_FREQUENCY * 2 * np.pi  # to rad/s
    energy_joules = energy * e * 1e6  # Convert to joules.
    voltage = (energy_joules / (e * angular_frequency * r34)) * gradient_m_per_s
    return voltage


def calculate_voltage(*, slope: float, r34: float, energy: float, frequency: float):
    # energy in MeV!!!
    # slope in m/s
    # r34 in m/rad
    # frequency in Hz
    energy_joules = energy * e * 1e6  # Convert to joules.
    angular_frequency = frequency * 2 * np.pi  # to rad/s
    voltage = (energy_joules / (e * angular_frequency * r34)) * slope
    return abs(voltage)

def r34s_from_scan(scan):
    result = []
    for measurement in scan:
        # Pick a non-bg image.
        im = measurement.images[0]
        result.append(r34_from_tds_to_screen(im.metadata))
    return np.array(result)
