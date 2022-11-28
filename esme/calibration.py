from typing import Any

import numpy as np
import pandas as pd
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from scipy.constants import e, c
from scipy.optimize import curve_fit

from esme.lattice import injector_cell_from_snapshot

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH



def line(x, a0, a1) -> Any:
    return a0 + a1 * x


class TDSCalibrator:
    def __init__(self, percentages, tds_slopes, dispersion,
                 tds_slope_units=None):
        self.percentages = np.array(percentages)
        self.tds_slopes = np.array(tds_slopes)
        self.dispersion = dispersion
        if tds_slope_units == "um/ps":
            self.tds_slopes = self.tds_slopes * 1e6
        elif tds_slope_units is not None:
            raise TypeError(f"Wrong tds_slope_units: {tds_slope_units}")

    def linear_fit(self):
        popt, pcov = curve_fit(line, self.percentages, self.tds_slopes)
        return popt, pcov

    def get_tds_slope(self, percentage):
        popt, _ = self.linear_fit()
        return line(percentage, *popt)

    def get_voltage(self, percentage, snapshot: pd.Series):
        tds_slope = self.get_tds_slope(percentage)
        return get_tds_voltage(tds_slope, snapshot)

    def __repr__(self):
        cname = type(self).__name__
        dx = self.dispersion
        return f"<{cname}: {dx=}, %={repr(self.percentage)}, grds={self.tds_slopes}>"

    # def get_voltage_for_dispersion_scan(percentage, snapshot: pd.Series)


class TrivialTDSCalibrator:
    def __init__(self, percentages, voltages):
        self.percentages = percentages
        self.voltages = voltages

    def get_voltage(self, percentage, snapshot):
        return dict(zip(self.percentages, self.voltages))[percentage]


def lat_from_tds_to_screen(snapshot: pd.Series):
    cell = injector_cell_from_snapshot(snapshot)
    screen_marker = next(ele for ele in cell if ele.id == "OTRC.64.I1D")
    tds_marker = next(ele for ele in cell if ele.id == "TDS2")

    lat = MagneticLattice(cell, start=tds_marker, stop=screen_marker)

    return lat


def r34_from_tds_to_screen(snapshot: pd.Series):
    lat = lat_from_tds_to_screen(snapshot)
    energy = snapshot[I1D_ENERGY_ADDRESS].mean()

    # Convert energy to GeV from MeV
    _, rmat, _ = lat.transfer_maps(energy * 1e-3)
    r34 = rmat[2, 3]
    return r34


def get_tds_voltage(gradient_m_per_s, snapshot: pd.Series):
    r34 = r34_from_tds_to_screen(snapshot)
    energy = snapshot[I1D_ENERGY_ADDRESS]
    angular_frequency = TDS_FREQUENCY * 2 * np.pi  # to rad/s
    energy_joules = energy * e * 1e6  # Convert to joules.
    voltage = (energy_joules / (e * angular_frequency * r34)) * gradient_m_per_s
    return voltage


# def get_tds_slope(voltage, snapshot: pd.Series):
#     from IPython import embed; embed()
#     r34 = r34_from_tds_to_screen(snapshot)
#     energy = snapshot[I1D_ENERGY_ADDRESS] * 1e-3
#     frequency = 3e9 * 2 * np.pi
#     tds_slope = voltage * e * frequency * r34 / energy
#     return tds_slope