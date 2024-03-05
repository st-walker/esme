import warnings
from typing import Optional
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.constants import c, e
from scipy.optimize import curve_fit

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from esme.maths import line
from esme import DiagnosticRegion
from esme.optics import calculate_i1d_r34_from_tds_centre

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH
TDS_LENGTH = 0.7  # metres



class AmplitudeVoltageMapping:
    def __init__(self, region: DiagnosticRegion, amplitudes, voltages):
        self.region = region
        self._amplitudes = amplitudes
        self._voltages = voltages

        self._amp_to_voltage_popt = self.fit_to_voltage()
        self._voltage_to_amp_popt = self.fit_to_amplitude()

    def get_voltage(self, amplitude):
        popt, _ = self._voltage_to_amp_popt
        return line(np.array(amplitude), *popt)

    def get_amplitude(self, voltage):
        popt, _ = self._amp_to_voltage_popt
        return line(np.array(voltage), *popt)

    def fit_to_voltage(self):
        popt, pcov = curve_fit(line, self._amplitudes, self._voltages)
        return popt, 
    def fit_to_amplitude(self):
        popt, pcov = curve_fit(line, self._voltages, self._amplitudes)
        return popt, pcov

    def __call__(self, amplitude):
        return self.get_amplitude(amplitude)


@dataclass
class CalibrationOptics:
    energy: float
    magnets: Optional[dict[str, float]] = None
    r12_streaking: Optional[float] = None
    frequency: Optional[float] = 3e9


class CompleteCalibration:
    CAL_M_PER_PS = 1e-12
    CAL_UM_PER_PS = 1e-6

    def __init__(self, region: DiagnosticRegion,
                 screen_name: str,
                 optics: CalibrationOptics,
                 amplitudes: list[float],
                 cal_factors: Optional[list[float]] = None,
                 voltages: Optional[list[float]] = None):

        self.region = region
        self.screen_name = screen_name
        self.optics = optics
        self.amplitudes = amplitudes
        self.cal_factors = cal_factors

        if cal_factors is None and voltages is not None:
            self.cal_factors = get_tds_com_slope(optics.r12_streaking,
                                                 optics.energy,
                                                 np.array(voltages))

        if cal_factors is None and voltages is None:
            raise ValueError("")

    def r34_from_optics(self):
        from IPython import embed; embed()
        return -5.5 #XXXXXX??

    def calculate_voltages(self):
        from IPython import embed; embed()

    def mapping(self) -> AmplitudeVoltageMapping:
        amps = self.amplitudes
        voltages = self.calculate_voltages(amps)
        return AmplitudeVoltageMapping(amps, voltages)


    # def r34(self):
    #     if not




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
