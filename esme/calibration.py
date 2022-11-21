import numpy as np
import pandas as pd
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from scipy.constants import e

from esme.lattice import injector_cell_from_snapshot


def lat_from_tds_to_screen(snapshot: pd.Series):
    cell = injector_cell_from_snapshot(snapshot)
    screen_marker = next(ele for ele in cell if ele.id == "OTRC.64.I1D")
    tds_marker = next(ele for ele in cell if ele.id == "TDS2")

    lat = MagneticLattice(cell, start=tds_marker, stop=screen_marker)

    return lat


def r34_from_tds_to_screen(snapshot: pd.Series):
    lat = lat_from_tds_to_screen(snapshot)
    energy = snapshot["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"].mean()

    # Convert energy to GeV from MeV
    _, rmat, _ = lat.transfer_maps(energy * 1e-3)
    r34 = rmat[2, 3]
    return r34


def get_tds_voltage(gradient_m_per_s, snapshot: pd.Series):
    r34 = r34_from_tds_to_screen(snapshot)
    energy = snapshot["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"]
    frequency = 3e9 * 2 * np.pi

    energy_joules = energy * e * 1e6  # Convert to joules.

    voltage = (energy_joules / (e * frequency * r34)) * gradient_m_per_s

    return voltage
