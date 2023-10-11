"""Main script for describing measurements (consisting of TDS voltage
scans and dispersion scans, as well as possible beta scans).  The full
set of measurements is reified with the SliceEnergySpreadMeasurement
class.  Each instance has at most one DispersionScan, at most one
TDSScan and at most one BetaScan instance, each class corresponding to
that scan.  Each scan consists of multiple measurements, represented
by ScanMeasurement instances.  Each ScanMeasurement consists of zero
or more TDSScreenImage instance, each of which can either be a
background image or a beam image (checked by calling is_bg on the
instance).

To get "the result", populate a SliceEnergySpreadMeasurement instance
with scans and call the all_fit_parameters method.  Probably needs
both a dispersion scan and a tds scan, but the beta scan is always
optional (but provides additional parameters in the final result).

Units: everything is in SI, except energy which is in eV.

"""

from __future__ import annotations

import logging
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.constants import c, e, m_e
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt  # pylint: disable=no-name-in-module
from scipy.constants import e

from esme.calibration import TDS_LENGTH, TDS_WAVENUMBER, TrivialTDSCalibrator
from esme.exceptions import EnergySpreadCalculationError, TDSCalibrationError
from esme.image import get_slice_properties, get_central_slice_width_from_slice_properties, filter_image
from esme.maths import ValueWithErrorT, linear_fit
import numpy as np
from esme.optics import calculate_i1d_r34_from_tds_centre


from esme.calibration import TDS_WAVENUMBER
from esme.maths import get_gaussian_fit


PIXEL_SCALE_X_UM: float = 13.7369
PIXEL_SCALE_Y_UM: float = 11.1756


PIXEL_SCALE_X_M: float = PIXEL_SCALE_X_UM * 1e-6
PIXEL_SCALE_Y_M: float = PIXEL_SCALE_Y_UM * 1e-6

LOG = logging.getLogger(__name__)

ELECTRON_MASS_EV: float = m_e * c**2 / e

RawImageT = npt.NDArray



class MeasurementDataFrames:
    def __init__(self, *, dscans, tscans, bscans):
        self.dscans = dscans
        self.tscans = tscans
        self.bscans = bscans

    @classmethod
    def from_filenames(cls, fnames, image_dir, image_address, energy_address):
        imd = image_dir
        ia = image_address
        ea = energy_address
        dscans = []
        tscans = []
        bscans = []
        for f in fnames:
            if "tscan" in str(f):
                tscans.append(SetpointDataFrame(pd.read_pickle(f),
                                                images_dir=imd,
                                                image_address=ia,
                                                energy_address=ea))
            elif "dscan" in str(f):
                dscans.append(SetpointDataFrame(pd.read_pickle(f),
                                                images_dir=imd,
                                                image_address=ia,
                                                energy_address=ea))
            elif "bscan" in str(f):
                bscans.append(SetpointDataFrame(pd.read_pickle(f),
                                                images_dir=imd,
                                                image_address=ia,
                                                energy_address=ea))
            else:
                raise ValueError(f"Unrecognised file: {f}")
        return cls(dscans=dscans, bscans=bscans, tscans=tscans)
        

    def max_voltage_df(self):
        imax = np.argmax([df.voltage for df in self.tscans])
        return self.tscans[imax]

    def max_dispersion_sp(self):
        imax = np.argmax([df.dispersion for df in self.tscans])
        return self.dscans[imax]
    
    def dscan_voltage(self):
        dscan_voltages = [dscan.voltage for dscan in self.dscans]
        assert len(set(dscan_voltages)) == 1
        return dscan_voltages[0]

    def tscan_dispersion(self):
        tscan_dispersions = [tscan.dispersion for tscan in self.tscans]
        assert len(set(tscan_dispersions)) == 1
        return tscan_dispersions[0]

    def energy(self):
        energies = []
        energies.extend([t.energy for t in self.tscans])
        energies.extend([b.energy for b in self.bscans])
        energies.extend([d.energy for d in self.dscans])
        return np.mean(energies)


class SetpointDataFrame:
    def __init__(self, df, *, images_dir, image_address, energy_address):
        self.df = df
        self.images_dir = Path(images_dir)
        self.image_address = image_address
        self.energy_address = energy_address

    @property
    def image_full_paths(self):
        paths = Path(self.images_dir) / self.df[self.image_address]
        return [p.resolve() for p in paths]

    @property
    def scan_type(self):
        return _get_constant_column_from_df(self.df, "scan_type")

    @property
    def dispersion(self):
        return _get_constant_column_from_df(self.df, "dispersion")

    @property
    def beta(self):
        return _get_constant_column_from_df(self.df, "beta")

    @property
    def voltage(self):
        return _get_constant_column_from_df(self.df, "voltage")

    @property
    def energy(self):
        """energy at the screen"""
        return np.mean(self.df[self.energy_address])

    @property
    def screen_name(self):
        return self.image_address.split("/")[2]

    def __repr__(self):
        v = self.voltage / 1e6
        d = self.dispersion
        return f"<{type(self).__name__}, {self.scan_type}, V={v}MV, D={d}m, β={self.beta}m>"

def _get_constant_column_from_df(df, name):
    col = df[name]
    unique_values = set(col)
    if len(unique_values) != 1:
        raise ValueError(f"Column {name} is not constant in dataframe")
    return unique_values.pop()


def transform_pixel_widths(
    pixel_widths: npt.NDArray,
    pixel_widths_errors: npt.NDArray,
    *,
    pixel_units: str = "px",
    to_variances: bool = True,
    dimension: str = "x",
) -> tuple[npt.NDArray, npt.NDArray]:
    """The fits used in the paper are linear relationships between the variances
    (i.e. pixel_std^2) and the square of the independent variable (either
    voltage squared V^2 or dipsersion D^2). This function takes D or V and sigma
    and transforms these variables to D^2 or V^2 and pixel width *variances* in
    units of um, so that the resulting fit for the standard deviations will be
    linear in the independent variable.

    """
    # Convert to ufloats momentarily for the error propagation.
    widths = np.array(
        [ufloat(v, e) for (v, e) in zip(pixel_widths, pixel_widths_errors)]
    )

    if pixel_units == "px":
        scale = 1
    elif pixel_units == "um" and dimension == "x":
        scale = PIXEL_SCALE_X_UM
    elif pixel_units == "um" and dimension == "y":
        scale = PIXEL_SCALE_Y_UM
    elif pixel_units == "m" and dimension == "x":
        scale = PIXEL_SCALE_X_M
    elif pixel_units == "m" and dimension == "y":
        scale = PIXEL_SCALE_Y_M
    else:
        raise ValueError(f"unknown unit or dimension: {pixel_units=}, {dimension=}")

    widths *= scale
    if to_variances:
        widths = widths**2

    # Extract errors form ufloats to go back to a 2-tuple of arrays.
    widths, errors = zip(*[(w.nominal_value, w.std_dev) for w in widths])
    return np.array(widths), np.array(errors)


def calculate_energy_spread_simple(widths) -> ValueWithErrorT:
    # Get measurement instance with highest dispresion for this scan
    dx, measurement = max((measurement.dx, measurement) for measurement in scan)
    energy = measurement.energy * 1e6 # in eV

    width_pixels = measurement.mean_central_slice_width_with_error(padding=10)
    # Calculate with uncertainties automatically.
    width_pixels_unc = ufloat(*width_pixels)
    width_unc = width_pixels_unc * PIXEL_SCALE_X_M

    energy_spread_unc = energy * width_unc / dx
    energy_spread_ev = energy_spread_unc

    value, error = energy_spread_ev.n, energy_spread_ev.s

    return value, error  # in eV


def pixel_widths_from_setpoint(setpoint: SetpointDataFrame):
    image_full_paths = setpoint.image_full_paths
    central_sigmas = []
    for path in image_full_paths:
        image = np.load(path)["image"]
        image = image.T # Flip to match control room..?  TODO
        # XXX: I do not use any bg here...
        image = filter_image(image, bg=0.0, crop=False)
        _, means, bunch_sigmas = get_slice_properties(image)
        sigma = get_central_slice_width_from_slice_properties(
            means, bunch_sigmas, padding=10
        )
        central_width_row = np.argmin(means)
        central_sigmas.append(sigma)

    return np.mean(central_sigmas)


@dataclass
class OpticsFixedPoints:
    beta_screen: float
    beta_tds: float
    alpha_tds: float

    @property
    def gamma_tds(self) -> float:
        return (1 + self.alpha_tds**2) / self.beta_tds


class SliceWidthsFitter:
    def __init__(
        self,
            dscan_widths,
            tscan_widths,
            bscan_widths=None,
    ):
        self.dscan_widths = dscan_widths
        self.tscan_widths = tscan_widths
        self.bscan_widths = bscan_widths

    def dispersion_scan_fit(self, sigma_r=None, emittance=None) -> ValueWithErrorT:
        widths_with_errors = list(self.dscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        dx = np.array(list(self.dscan_widths.keys()))
        dx2 = dx**2

        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v

    def tds_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths_with_errors = list(self.tscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        voltages = np.array(list(self.tscan_widths.keys()))
        voltages2 = voltages**2
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v

    def beta_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths_with_errors = list(self.bscan_widths.values())
        widths = [x.n for x in widths_with_errors]
        errors = [x.s for x in widths_with_errors]
        beta = np.array(list(self.bscan_widths.keys()))
        widths2_m2, errors2_m2 = transform_pixel_widths(
            widths, errors, pixel_units="m", to_variances=True
        )
        a_beta, b_beta = linear_fit(beta, widths2_m2, errors2_m2)
        return a_beta, b_beta

    def all_fit_parameters(self, beam_energy, tscan_dispersion, dscan_voltage, optics_fixed_points, sigma_r=None, emit=None) -> FittedBeamParameters:
        a_v, b_v = self.tds_scan_fit()
        a_d, b_d = self.dispersion_scan_fit()

        # Values and errors, here we just say there is 0 error in the
        # dispersion, voltage and energy, not strictly true of course.
        energy = beam_energy, 0.0  # in eV
        dispersion = tscan_dispersion, 0.0
        voltage = dscan_voltage, 0.0

        try:
            a_beta, b_beta = self.beta_scan_fit()
        except (TypeError, AttributeError):
            a_beta = b_beta = None

        return FittedBeamParameters(
            a_v=a_v,
            b_v=b_v,
            a_d=a_d,
            b_d=b_d,
            reference_energy=energy,
            reference_dispersion=dispersion,
            reference_voltage=voltage,
            oconfig=optics_fixed_points,
            a_beta=a_beta,
            b_beta=b_beta,
        )


# @dataclass
# class ScanSetpoint:
#     dispersion: float


# @dataclass
# class ReducedFittedParameters:


@dataclass
class FittedBeamParameters:
    """Table 2 from the paper"""

    # Stored as tuples, nominal value with error.
    a_v: ValueWithErrorT
    b_v: ValueWithErrorT
    a_d: ValueWithErrorT
    b_d: ValueWithErrorT
    reference_energy: ValueWithErrorT
    reference_dispersion: ValueWithErrorT
    reference_voltage: ValueWithErrorT
    oconfig: OpticalConfig
    a_beta: ValueWithErrorT = None
    b_beta: ValueWithErrorT = None

    def a_d_derived(self, sigma_r, emittance=None):
        if emittance is None:
            emittance = self.emitx
        return np.sqrt(sigma_r**2 + (emittance * self.oconfig.beta_screen)**2)

    def set_a_d_to_known_value(self, sigma_r, emittance=None):
        new_ad = self.a_d_derived(sigma_r, emittance=emittance)
        self.a_d = new_ad

    def set_a_v_to_known_value(self, sigma_r, emittance=None):
        new_ad = self.a_d_derived(sigma_r, emittance=emittance)
        self.a_d = new_ad


    @property
    def sigma_e(self) -> ValueWithErrorT:
        energy0 = ufloat(*self.reference_energy)
        dx0 = ufloat(*self.reference_dispersion)
        # Convert to ufloat for correct error propagation before converting back
        # to tuples at the end.
        av = ufloat(*self.a_v)
        ad = ufloat(*self.a_d)
        result = (energy0 / dx0) * usqrt(av - ad)
        return result.n, result.s

    @property
    def sigma_e_alt(self) -> ValueWithErrorT:
        energy0 = ufloat(*self.reference_energy)
        dx0 = ufloat(*self.reference_dispersion)
        v0 = ufloat(*self.reference_voltage)
        # Convert to ufloat for correct error propagation before converting back
        # to tuples at the end.
        bd = ufloat(*self.b_d)
        bv = ufloat(*self.b_v)
        result = (energy0 / dx0) * usqrt(bd * dx0**2 - bv * v0**2)
        return result.n, result.s

    @property
    def sigma_i(self) -> ValueWithErrorT:
        """This is the average beamsize in the TDS, returned in metres"""
        k = TDS_WAVENUMBER
        dx0 = ufloat(*self.reference_dispersion)
        energy0 = ufloat(*self.reference_energy)
        e0_joules = energy0 * e
        bv = ufloat(*self.b_v)
        try:
            result = (e0_joules / (dx0 * e * k)) * usqrt(bv)
        except ValueError:
            return np.nan, np.nan
        return result.n, result.s

    @property
    def sigma_i_alt(self) -> ValueWithErrorT:
        """This is the average beamsize in the TDS, returned in metres"""
        av = ufloat(*self.a_v)
        ad = ufloat(*self.a_d)
        bd = ufloat(*self.b_d)

        dx0 = ufloat(*self.reference_dispersion)
        v0 = abs(ufloat(*self.reference_voltage))
        e0j = ufloat(*self.reference_energy) * e  # Convert to joules
        k = TDS_WAVENUMBER
        try:
            result = (e0j / (dx0 * e * k * v0)) * usqrt(ad - av + dx0**2 * bd)
        except ValueError:
            return np.nan, np.nan
        return result.n, result.s

    @property
    def sigma_b(self) -> ValueWithErrorT:
        bety = self.oconfig.beta_tds
        alfy = self.oconfig.alpha_tds
        gamy = self.oconfig.gamma_tds
        length = TDS_LENGTH
        sigma_i = ufloat(*self.sigma_i)
        b_beta = sigma_i**2 / (bety + 0.25 * length**2 * gamy - length * alfy)
        result = usqrt(b_beta * self.oconfig.beta_screen)
        return result.n, result.s

    @property
    def sigma_r(self) -> ValueWithErrorT:
        # if self.known_sigma_r:
        #     return self.known_sigma_r, 0
        ad = ufloat(*self.a_d)
        sigma_b = ufloat(*self.sigma_b)
        result = usqrt(ad - sigma_b**2)
        return result.n, result.s

    @property
    def emitx(self) -> ValueWithErrorT:
        gam0 = ufloat(*self.reference_energy) / ELECTRON_MASS_EV
        sigma_b = ufloat(*self.sigma_b)
        result = sigma_b**2 * gam0 / self.oconfig.beta_screen
        return result.n, result.s

    @property
    def sigma_b_alt(self) -> ValueWithErrorT:
        ab = ufloat(*self.a_beta)
        ad = ufloat(*self.a_d)
        bd = ufloat(*self.b_d)
        d0 = ufloat(*self.reference_dispersion)
        result = usqrt(ad + bd * d0**2 - ab)
        return result.n, result.s

    @property
    def emitx_alt(self) -> ValueWithErrorT:
        bb = ufloat(*self.b_beta)
        gamma0 = ufloat(*self.reference_energy) / ELECTRON_MASS_EV
        result = bb * gamma0
        return result.n, result.s

    @property
    def sigma_r_alt(self) -> ValueWithErrorT:
        # if self.known_sigma_r:
        #     return self.known_sigma_r, 0
        ab = ufloat(*self.a_beta)
        bd = ufloat(*self.b_d)
        d0 = ufloat(*self.reference_dispersion)
        result = usqrt(ab - bd * d0**2)
        return result.n, result.s

    @property
    def sigma_e_from_tds(self) -> ValueWithErrorT:
        sigma_i = ufloat(*self.sigma_i)
        in_ev = abs(ufloat(*self.reference_voltage)) * TDS_WAVENUMBER * sigma_i
        return in_ev.n, in_ev.s

    @property
    def sigma_e_from_tds_alt(self) -> ValueWithErrorT:
        sigma_i = ufloat(*self.sigma_i_alt)
        in_ev = abs(ufloat(*self.reference_voltage)) * TDS_WAVENUMBER * sigma_i
        return in_ev.n, in_ev.s

    def fit_parameters_to_df(self) -> pd.DataFrame:
        dx0 = self.reference_dispersion
        v0 = self.reference_voltage
        e0 = self.reference_energy

        av, bv = self.a_v, self.b_v
        ad, bd = self.a_d, self.b_d

        pdict = {
            "V_0": v0,
            "D_0": dx0,
            "E_0": e0,
            "A_V": av,
            "B_V": bv,
            "A_D": ad,
            "B_D": bd,
        }

        if self.a_beta and self.b_beta:
            pdict |= {"A_beta": self.a_beta, "B_beta": self.b_beta}

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _beam_parameters_to_df(self) -> pd.DataFrame:
        pdict = {
            "sigma_e": self.sigma_e,
            "sigma_i": self.sigma_i,
            "sigma_e_from_tds": self.sigma_e_from_tds,
            "sigma_b": self.sigma_b,
            "sigma_r": self.sigma_r,
            "emitx": self.emitx,
        }

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _alt_beam_parameters_to_df(self) -> pd.DataFrame:
        pdict = {
            "sigma_e": self.sigma_e_alt,
            "sigma_i": self.sigma_i_alt,
            "sigma_e_from_tds": self.sigma_e_from_tds_alt,
        }
        if self.a_beta and self.b_beta:
            pdict |= {
                "sigma_b": self.sigma_b_alt,
                "sigma_r": self.sigma_r_alt,
                "emitx": self.emitx_alt,
            }
        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame(
            {"alt_values": values, "alt_errors": errors}, index=pdict.keys()
        )

    def beam_parameters_to_df(self, sigma_z=None) -> pd.DataFrame:
        # sigma_t in picosecondsm
        params = self._beam_parameters_to_df()
        alt_params = self._alt_beam_parameters_to_df()
        if sigma_z is not None:
            sigma_t = sigma_z / c
            params.loc["sigma_z"] = {"values": sigma_z.n, "errors": sigma_z.s}
            params.loc["sigma_t"] = {"values": sigma_t.n, "errors": sigma_t.s}

        return pd.concat([params, alt_params], axis=1)


def true_bunch_length_from_df(setpoint):
    image_full_paths = setpoint.image_full_paths

    apparent_bunch_lengths = []
    for path in image_full_paths:
        image = np.load(path)["image"]
        image = image.T # Flip to match control room..?  TODO
        # XXX: I do not use any bg here...
        image = filter_image(image, bg=0.0, crop=False)
        length, error = apparent_bunch_length_from_processed_image(image)
        apparent_bunch_lengths.append(ufloat(length, error))
        
    mean_apparent_bunch_length = np.mean(apparent_bunch_lengths)

    r34 = abs(calculate_i1d_r34_from_tds_centre(setpoint.df, setpoint.screen_name, setpoint.energy))
    energy = setpoint.energy * 1e6 * e # to Joules
    voltage = setpoint.voltage
    bunch_length = (energy / (e * voltage * TDS_WAVENUMBER)) * mean_apparent_bunch_length / r34

    return bunch_length


def apparent_bunch_length_from_processed_image(image):
    pixel_indices = np.arange(image.shape[0])  # Assumes streaking is in image Y
    projection = image.sum(axis=1)
    popt, perr = get_gaussian_fit(pixel_indices, projection)
    sigma = popt[2]
    sigma_error = perr[2]

    # Transform units from px to whatever was chosen
    mean_length, mean_error = transform_pixel_widths(
        [sigma],
        [sigma_error],
        to_variances=False,
        pixel_units="m",
        dimension="y",
    )

    return np.squeeze(mean_length), np.squeeze(mean_error)

def true_bunch_length_from_processed_image(image, *, voltage, r34, energy) -> ufloat:
    raw_bl, raw_bl_err = apparent_bunch_length_from_processed_image(image)
    raw_bl = ufloat(raw_bl, raw_bl_err)
    true_bl = (energy / (e * voltage * TDS_WAVENUMBER)) * raw_bl / abs(r34)
    return true_bl


def _mean_with_uncertainties(values, stdevs):
    # Calculate mean of n values each with 1 uncertainty.
    assert len(values) == len(stdevs)
    mean = np.mean(values)
    variances = np.power(stdevs, 2)
    mean_stdev = np.sqrt(np.sum(variances) / (len(values) ** 2))
    return mean, mean_stdev
