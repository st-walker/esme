from typing import Any

import numpy as np
import pandas as pd
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from scipy.constants import e, c
from scipy.optimize import curve_fit

from esme.lattice import injector_cell_from_snapshot
import esme.analysis as ana

I1D_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

TDS_FREQUENCY = 3e9
TDS_WAVELENGTH = c / TDS_FREQUENCY
TDS_WAVENUMBER = 2 * np.pi / TDS_WAVELENGTH



def line(x, a0, a1) -> Any:
    return a0 + a1 * x


def sqrt(x, a0, a1) -> Any:
    return a0 + a1 * np.sqrt(x)


class TDSCalibrator:
    def __init__(self, percentages, tds_slopes, dispersion,
                 tds_slope_units=None, fn=line):
        self.percentages = np.array(percentages)
        self.tds_slopes = np.array(tds_slopes)
        self.dispersion = dispersion
        if tds_slope_units == "um/ps":
            self.tds_slopes = self.tds_slopes * 1e6
        elif tds_slope_units is not None:
            raise TypeError(f"Wrong tds_slope_units: {tds_slope_units}")
        self.fn = fn

    def fit(self):
        popt, pcov = curve_fit(self.fn, self.percentages, self.tds_slopes)
        return popt, pcov

    def get_tds_slope(self, percentage):
        popt, _ = self.fit()
        return self.fn(percentage, *popt)

    def get_voltage(self, percentage, snapshot: pd.Series):
        tds_slope = self.get_tds_slope(percentage)
        return get_tds_voltage(tds_slope, snapshot)

    def __repr__(self):
        cname = type(self).__name__
        dx = self.dispersion
        return f"<{cname}: {dx=}, %={repr(self.percentages)}, grds={self.tds_slopes}>"

    # def get_voltage_for_dispersion_scan(percentage, snapshot: pd.Series)


class TrivialTDSCalibrator:
    def __init__(self, percentages, voltages, dispersion):
        self.percentages = np.array(percentages)
        self.voltages = np.array(voltages)
        self.dispersion = np.array(dispersion)

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


def r34s_from_scan(scan: ana.ParameterScan):
    result = []
    for measurement in scan:
        # Pick a non-bg image.
        im = measurement.images[0]
        result.append(r34_from_tds_to_screen(im.metadata))
    return np.array(result)
