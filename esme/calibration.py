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
from esme import DiagnosticRegion

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH
TDS_LENGTH = 0.7  # metres



class TDSCalibration:
    def __init__(self, region: DiagnosticRegion, modulator_voltage=None):
        self.region = region
        self.modulator_voltage = modulator_voltage

    def get_voltage(self, amplitude):
        popt, _ = self.fit_to_voltage()
        return line(amplitude, *popt)

    def get_amplitude(self, voltage):
        popt, _ = self.fit_to_amplitude()
        return line(voltage, *popt)

    def fit_to_voltage(self):
        popt, pcov = curve_fit(line, self.get_amplitudes(), self.get_voltages())
        return popt, pcov

    def fit_to_amplitude(self):
        popt, pcov = curve_fit(line, self.get_voltages(), self.get_amplitudes())
        return popt, pcov
    

class StuartCalibration(TDSCalibration):
    def __init__(self, region: DiagnosticRegion, amplitudes, voltages, modulator_voltage=None):
        super().__init__(region, modulator_voltage=modulator_voltage)
        self.amplitudes = amplitudes
        self.voltages = voltages

    def get_voltages(self):
        return self.voltages

    def get_amplitudes(self):
        return self.amplitudes


class BolkoCalibrationSetpoint:
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
    def __init__(self, region: DiagnosticRegion,
                 bolko_setpoints: list[BolkoCalibrationSetpoint],
                 modulator_voltage=None):
        super().__init__(region, modulator_voltage=modulator_voltage)
        self.setpoints = bolko_setpoints

    def get_amplitudes(self):
        return np.array([setpoint.amplitude for setpoint in self.setpoints])

    def get_voltage(self):
        return np.array([setpoint.get_voltage() for setpoint in self.setpoints])


class IgorCalibration(TDSCalibration):
    def __init__(self, region: DiagnosticRegion, amplitudes, voltages):
        super().__init__(region)
        self.amplitudes = amplitudes
        self.voltages = voltages

    def get_amplitude(self, voltage):
        return dict(zip(self.voltages, self.amplitudes))[voltage]

    def get_voltage(self, amplitude):
        return dict(zip(self.amplitudes, self.voltages))[amplitude]


class DiscreteCalibration(TDSCalibration):
    def __init__(self, region: DiagnosticRegion, amplitudes, voltages):
        super().__init__(region)
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

def get_tds_com_slope(r12_streaking, energy_mev, voltage) -> float:
    """Calculate TDS calibration / TDS centre of mass slope.
    Energy in MeV
    frequency in cycles/s
    voltage in V

    """

    angular_frequency = TDS_FREQUENCY * 2 * np.pi  # to rad/s
    energy_joules = energy_mev * e * 1e6  # Convert to joules.
    gradient_m_per_s = e * voltage * r12_streaking * angular_frequency / energy_joules
    return gradient_m_per_s
