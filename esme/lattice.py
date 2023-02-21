"""Used for calibrating the measurement and also deriving measurement."""

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

from ocelot.utils.fel_track import SectionedFEL, FELSection
from ocelot.cpbd.elements import Quadrupole

from esme.measurement import QuadrupoleSetting
from .sections import sections

LOG = logging.getLogger(__name__)


def design_quad_strengths() -> pd.DataFrame:
    # This is all very hardcoded for the measurement we did in October 2022.
    # Get data in correct shape for passing to pd.DataFrame
    # I think these are what they're "supposed" to be, should really check...
    dispersions = [1.197, 1.025, 0.801, 0.6590]
    # These are quads that are set at the start of the measurement and then left
    # constant for the whole scan.

    upstream_quad_names = [
        "QI.52.I1",
        "QI.53.I1",
        "QI.54.I1",
        "QI.55.I1",
        "QI.57.I1",
        "QI.59.I1",
        # "QI.60.I1", "QI.61.I1", "QI.63.I1D", "QI.64.I1D"
    ]
    # These are K1Ls but in mrad/m. whereas k1l is "usually" rad/m. These are
    # the upstream/matching quads and stay the same for the whole scan.
    upstream_strengths = [
        -83.7203,
        500.308,
        188.82,
        -712.48,
        712.48,
        -712.48,
        # Commented out as they are set below in scan_strengths.
        # -509.559, 832.63, -249.585, 831.95
    ]

    # These are the quads which change once per point in the dispersion scan.
    scan_quad_names = ["QI.60.I1", "QI.61.I1", "QI.63.I1D", "QI.64.I1D"]
    scan_strengths = [
        [-509.559, 832.63, -249.585, 831.95],
        [-509.0345, 832.175, 106.965, 475.4],
        [-508.965, 820.2076, 582.365, 0],
        [-508.749, 789.625, 1046.306, -475.4],
    ]

    result = []

    for dx, strengths in zip(dispersions, scan_strengths):
        strengths = upstream_strengths + strengths
        stds = np.zeros_like(strengths)
        quad_names = upstream_quad_names + scan_quad_names
        df = pd.DataFrame(np.vstack((strengths, stds)).T, index=quad_names, columns=["kick_mean", "kick_std"])

        df = _append_element_positions_and_names(df, quad_names)
        result.append(df)

    return dispersions, result


def _append_element_positions_and_names(df: pd.DataFrame, names: Iterable[str]):
    cell = cell_to_injector_dump()
    lengths = []
    positions = []
    for name in names:
        try:
            length = cell.element_attributes(name, "l")
        except KeyError:
            length = np.nan
        try:
            s = cell.element_s(name)
        except KeyError:
            s = np.nan

        lengths.append(length)
        positions.append(s)
    return df.assign(length=lengths, s=positions)


def mean_quad_strengths(df: pd.DataFrame, include_s=True, dropna=True):
    pattern = re.compile(r"Q(I?)|(L[NS])\.")
    quad_names = [key for key in df.keys() if pattern.match(key)]

    df_quads = df[quad_names]

    df = pd.DataFrame({"kick_mean": df_quads.mean(), "kick_std": df_quads.std()})

    df = _append_element_positions_and_names(df, quad_names)

    if dropna:
        return df.dropna()
    return df


def cell_to_injector_dump():
    return make_i1_cell() + make_i1d_cell()

def cell_to_bc2_dump():
    return make_i1_cell() + make_l1_cell() + make_l2_cell() + make_b2d_cell()

def injector_cell_from_snapshot(snapshot: pd.Series, check=True, change_correctors=False):
    quad_mask = snapshot.index.str.contains(r"^Q(?:I?|L[NS])\.")
    solenoid_mask = snapshot.index.str.contains(r"^SOL[AB]\.")
    corrector_mask = snapshot.index.str.contains(r"^C[IK]?[XY]\.")
    bends_mask = snapshot.index.str.contains(r"^B[BL]\.")
    used_mask = quad_mask | solenoid_mask | corrector_mask | corrector_mask | bends_mask

    if check:
        xfel_mask = snapshot.index.str.startswith("XFEL.")
        bpm_mask = snapshot.index.str.contains(r"^BPM[GAFRSCD]\.")
        timestamp_mask = snapshot.index == "timestamp"
        sextupole_mask = snapshot.index.str.contains(r"^SC\.")
        other_kickers_mask = snapshot.index.str.contains(r"CB[LB]\.")
        unused_mask = xfel_mask | bpm_mask | timestamp_mask | sextupole_mask | other_kickers_mask
        my_added_metadata = snapshot.index.str.startswith("MY_")
        the_rest_mask = ~(used_mask | unused_mask | my_added_metadata)
        snapshot[the_rest_mask]

        the_rest = snapshot[the_rest_mask]

        expected = {"RF.23.I1", "BK.24.I1"}
        if set(the_rest.index) != expected:
            LOG.warning("Unexpected item in DF.  Expected: {expected}, got: {the_rest}")

    cell = cell_to_injector_dump()
    for quad_name, int_strength in snapshot[quad_mask].items():
        try:
            quad = cell[quad_name]
        except KeyError:
            LOG.debug("NOT setting quad strength from snapshot: %s", quad_name)
        else:
            k1 = int_strength * 1e-3 / quad.l  # Convert to rad/m from mrad/m then to k1.
            LOG.debug(f"setting quad strength from snapshot: {quad_name=}, {quad.k1=} to {k1=}")
            quad.k1 = k1

    dipole_mask = bends_mask
    if change_correctors:
        dipole_mask |= corrector_mask
    for dipole_name, angle_mrad in snapshot[dipole_mask].items():
        try:
            dipole = cell[dipole_name]
        except KeyError:
            LOG.debug("NOT setting dipole angle from snapshot: %s", dipole_name)
        else:
            # I insert a negative sign here.  I guess convention is opposite?
            angle = -angle_mrad * 1e-3
            LOG.debug(f"setting dipole angle from snapshot: {dipole_name=}, {dipole.angle=} to {angle=}")
            dipole.angle = angle

    return cell


def apply_quad_setting_to_lattice(lattice: SectionedFEL, qset: QuadrupoleSetting):
    LOG.debug("Applying quadrupole strengths from QuadrupoleSetting instance to OCELOT SectionLattice.")
    for section in lattice.sections:
        sequence = section.lattice.sequence
        for element in sequence:
            if not isinstance(element, Quadrupole):
                continue
            try:
                k1l = qset.k1l_from_name(element.id)
            except ValueError:
                continue

            LOG.debug(f"{element.id} k1 before: {element.k1}")
            element.k1 = k1l / element.l
            LOG.debug(f"{element.id} k1 after: {element.k1}")


def make_to_i1d_lattice(data_dir="./"):
    all_sections = [sections.G1, sections.A1, sections.AH1, sections.LH, sections.I1D]
    return SectionedFEL(all_sections, data_dir=data_dir)

def make_to_b2d_lattice(data_dir="./"):
    all_sections = [sections.G1, sections.A1, sections.AH1, sections.LH, sections.DL, sections.BC0, sections.L1, sections.BC1, sections.L2, sections.BC2, sections.B2D]
    return SectionedFEL(all_sections, data_dir=data_dir)

def make_dummy_lookup_sequence():
    """Just make a cell of every single element for looking up
    strengths and lengths etc.  Obviously not for tracking, because
    it's not in the right order."""
    return sections.i1.make_cell() + sections.i1d.make_cell() + sections.l1.make_cell() + sections.l2.make_cell() + sections.b2d.make_cell()
