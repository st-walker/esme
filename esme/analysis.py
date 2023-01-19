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

import contextlib
import logging
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.constants import c, e, m_e
from uncertainties import ufloat, umath
from uncertainties.umath import sqrt as usqrt # pylint: disable=no-name-in-module

from esme.calibration import TDS_WAVENUMBER, TDS_LENGTH
from esme.maths import linear_fit, ValueWithErrorT
from esme.image import get_slice_properties, process_image
from esme.injector_channels import (TDS_AMPLITUDE_READBACK_ADDRESS,
                                    BEAM_ALLOWED_ADDRESS,
                                    BEAM_ENERGY_ADDRESS,
                                    EVENT10_CHANNEL,
                                    TDS_ON_BEAM_EVENT10,
                                    DUMP_SCREEN_ADDRESS)



PIXEL_SCALE_X_UM: float = 13.7369
PIXEL_SCALE_Y_UM: float = 11.1756


PIXEL_SCALE_X_M: float = PIXEL_SCALE_X_UM * 1e-6
PIXEL_SCALE_Y_M: float = PIXEL_SCALE_Y_UM * 1e-6

LOG = logging.getLogger(__name__)

ELECTRON_MASS_EV: float = m_e * c**2 / e

RawImageT = npt.NDArray

MULTIPROCESSING = True


class TDSScreenImage:
    def __init__(self, metadata):
        self.metadata = metadata
        self._image = None

    @property
    def filename(self) -> str:
        return self.metadata[DUMP_SCREEN_ADDRESS]

    def to_im(self, process=True) -> RawImageT:
        # path to png is in the df, but actually we want path to the adjacent
        # pcl file.  the pngs are just for debugging.
        fname = Path(self.filename).with_suffix(".pcl")
        with open(fname, "rb") as f:
            im = pickle.load(f)
        # Flip to match what we see in the control room.  not sure if
        # I need an additional flip here or not, but shouldn't matter too much.
        return im.T

    def show(self, raw=False) -> None:
        im = self.to_im(process=not raw)
        fig, ax = plt.subplots()
        ax.imshow(im)
        plt.show()

    @property
    def is_bg(self) -> bool:
        return not bool(self.metadata[BEAM_ALLOWED_ADDRESS])

    @property
    def beam_energy(self) -> float:
        eV = 1e6  # Energy is given in MeV which we convert to eV for consistency.
        return self.metadata[BEAM_ENERGY_ADDRESS] * eV

    @property
    def is_bad(self):
        beam_on = not self.is_bg
        tds_off = not self.tds_was_on
        return beam_on and tds_off

    @property
    def tds_was_on(self):
        return self.metadata[EVENT10_CHANNEL] == TDS_ON_BEAM_EVENT10

class ScanMeasurement:
    DF_DX_SCREEN_KEY = "MY_SCREEN_DX"
    DF_BETA_SCREEN_KEY = "MY_SCREEN_BETA"

    def __init__(self, df_path: os.PathLike):
        """bad_image_indices are SKIPPED and not loaded at all."""
        LOG.debug(f"Loading measurement: {df_path=}")
        df_path = Path(df_path)


        df = pd.read_pickle(df_path)

        self.dx = _get_constant_key_from_df_safeley(df, self.DF_DX_SCREEN_KEY)
        self.tds_percentage = df[TDS_AMPLITUDE_READBACK_ADDRESS].mean()

        try:
            self.beta = _get_constant_key_from_df_safeley(df, self.DF_BETA_SCREEN_KEY)
        except KeyError:
            pass

        self.images = []
        self.bg = []
        self._mean_bg_im: Optional[RawImageT] = None  # For caching.

        for i, relative_path in enumerate(df[DUMP_SCREEN_ADDRESS]):
            abs_path = df_path.parent / relative_path
            LOG.debug(f"Loading image index {i} @ {relative_path}")

            # In the df the png paths relative to the pickled dataframe, but
            # I want to be able to call it from anywhere, so resolve them to
            # absolute paths.
            abs_path = df_path.parent / relative_path
            metadata = df[df[DUMP_SCREEN_ADDRESS] == relative_path].squeeze()
            metadata[DUMP_SCREEN_ADDRESS] = abs_path
            image = TDSScreenImage(metadata)

            if image.is_bad:
                LOG.info(f"Skipping bad image: {i} (TDS was off whilst beam was on)")
                continue

            if image.is_bg:
                LOG.debug(f"Image {i} is bg")
                self.bg.append(image)
            elif not image.is_bg:
                self.images.append(image)

    @property
    def metadata(self) -> pd.Dataframe:
        df = pd.DataFrame([image.metadata for image in self.images])
        df = df.sort_values("timestamp")
        return df

    def to_im(self, index: int, process: bool = True) -> RawImageT:
        image = self[index].to_im()
        mean_bg = self.mean_bg_im()
        if process:
            image = process_image(image, mean_bg)
        return image

    def mean_bg_im(self) -> RawImageT:
        if self._mean_bg_im is not None:
            return self._mean_bg_im
        bgs = [tdsdata.to_im() for tdsdata in self.bg]
        self._mean_bg_im = np.mean(bgs, axis=0)
        return self._mean_bg_im

    def show(self, index: int) -> None:
        im = self.to_im(index)
        fig, ax = plt.subplots()
        ax.imshow(im)
        x, means, _ = get_slice_properties(im)
        means = [m.n for m in means]
        ax.plot(x, means, label="Slice positions")
        ax.axvline(x[np.argmin(means)], color="white", alpha=0.25)
        plt.show()

    def mean_central_slice_width_with_error(self, padding: int = 10) -> ValueWithErrorT:
        image_fitted_sigmas = []
        for i in range(self.nimages):
            image = self.to_im(i)
            # Get slice properties for this image
            x, means, sigmas = get_slice_properties(image)
            # Find highest energy slice (min because 0 is at the top in the image)
            centre_index = means.argmin()

            sigma = np.mean(sigmas[centre_index - padding : centre_index + padding])
            image_fitted_sigmas.append(sigma)

        width_with_error = np.mean(image_fitted_sigmas)

        LOG.debug(f"Calculated average slice width: {width_with_error}")

        return width_with_error.n, width_with_error.s  # To tuple

    def __getitem__(self, key: int) -> TDSScreenImage:
        return self.images[key]

    @property
    def nimages(self) -> int:
        return len(self.images)

    @property
    def beam_energy(self) -> float:
        return np.mean([im.beam_energy for im in self.images])

    def flatten(self, include_bg: bool = True) -> Generator[TDSScreenImage]:
        if include_bg:
            yield from self.bg
        yield from self.images

    def __repr__(self) -> str:
        tname = type(self).__name__
        nimages = len(self.images)
        nbg = len(self.bg)
        return f"<{tname}: Dx={self.dx} nimages = {nimages}, nbg={nbg}>"

def _f(measurement):
    return measurement.mean_central_slice_width_with_error(padding=10)


class ParameterScan:
    def __init__(
        self,
        files: Iterable[os.PathLike],
        calibrator=None,
    ):
        # Ideally this voltage, calibration etc. stuff should go in the df.
        # for now, whatever.
        LOG.debug(f"Instantiating {type(self).__name__}")
        self.calibrator = calibrator
        self.measurements = [ScanMeasurement(df_path) for df_path in files]

    @property
    def dx(self) -> npt.NDArray:
        return np.array([s.dx for s in self.measurements])

    @property
    def tds_percentage(self) -> npt.NDArray:
        return np.array([s.tds_percentage for s in self.measurements])

    @property
    def metadata(self) -> list[pd.Dataframe]:
        return [m.metadata for m in self.measurements]

    def max_energy_slice_widths_and_errors(
        self, padding: int = 20, do_mp: bool = True
    ) -> tuple[npt.NDArray, npt.NDArray]:
        global MULTIPROCESSING
        if MULTIPROCESSING and do_mp:
            with mp.Pool(mp.cpu_count()) as pool:
                widths_with_errors = np.array(pool.map(_f, self.measurements))
        else:
            widths_with_errors = np.array([_f(m) for m in self.measurements])

        widths = widths_with_errors[..., 0]
        errors = widths_with_errors[..., 1]
        return widths, errors

    def __getitem__(self, key: int) -> ScanMeasurement:
        return self.measurements[key]

    def beam_energy(self) -> float:
        return np.mean([m.beam_energy for m in self])

    def flatten(self, include_bg: bool = False) -> Generator[TDSScreenImage]:
        for measurement in self.measurements():
            yield from measurement.flatten(include_bg)

    def __iter__(self):
        return iter(self.measurements)


class DispersionScan(ParameterScan):
    @property
    def voltage(self):
        return _get_constant_voltage_for_scan(self)

class TDSScan(ParameterScan):
    @property
    def tds_slope(self):
        return np.array([s.tds_slope for s in self.measurements])

    @property
    def voltage(self):
        dx = self.dx
        # Get metadata associated with first (non-bg) image of each measurement,
        # and reasonably assume it's the same for every image of the scan.
        scan_metadata = [m.images[0].metadata for m in self.measurements]
        voltages = []
        for percentage, metadata in zip(self.tds_percentage, scan_metadata):
            voltages.append(self.calibrator.get_voltage(percentage, metadata))

        return np.array(voltages)


class BetaScan(ParameterScan):
    @property
    def beta(self):
        return np.array([s.beta for s in self.measurements])

    @property
    def voltage(self):
        return _get_constant_voltage_for_scan(self)



def transform_pixel_widths(
        pixel_widths, pixel_widths_errors, *, pixel_units="px", to_variances=True, dimension="x"
) -> tuple[npt.NDArray, npt.NDArray]:
    """The fits used in the paper are linear relationships between the variances
    (i.e. pixel_std^2) and the square of the independent variable (either
    voltage squared V^2 or dipsersion D^2). This function takes D or V and sigma
    and transforms these variables to D^2 or V^2 and pixel width *variances* in
    units of um, so that the resulting fit for the standard deviations will be
    linear in the independent variable.

    """
    # Convert to ufloats momentarily for the error propagation.
    widths = np.array([ufloat(v, e) for (v, e) in zip(pixel_widths, pixel_widths_errors)])

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


def calculate_energy_spread_simple(scan: DispersionScan) -> ValueWithErrorT:
    # Get measurement instance with highest dispresion for this scan
    dx, measurement = max((measurement.dx, measurement) for measurement in scan)
    energy = measurement.beam_energy  # in eV

    width_pixels = measurement.mean_central_slice_width_with_error(padding=10)
    # Calculate with uncertainties automatically.
    width_pixels_unc = ufloat(*width_pixels)
    width_unc = width_pixels_unc * PIXEL_SCALE_X_M

    energy_spread_unc = energy * width_unc / dx
    energy_spread_ev = energy_spread_unc

    value, error = energy_spread_ev.n, energy_spread_ev.s

    return value, error  # in eV


def _get_constant_key_from_df_safeley(df, key_name):
    col = df[key_name]
    value = col.iloc[0]
    if not (value == col).all():
        raise MalformedSnapshotDataFrame(f"{key_name} in {df} should be constant but is not")
    return value


@dataclass
class OpticalConfig:
    ocr_betx: float
    tds_bety: float
    tds_alfy: float

    @property
    def tds_gamy(self) -> float:
        return (1 + self.tds_alfy**2) / self.tds_bety


class SliceEnergySpreadMeasurement:
    def __init__(self, dscan: DispersionScan, tscan: TDSScan, optical_config: OpticalConfig, bscan: BetaScan = None):
        self.dscan = dscan
        self.tscan = tscan
        self.oconfig = optical_config
        self.bscan = bscan

    def dispersion_scan_fit(self) -> ValueWithErrorT:
        widths, errors = self.dscan.max_energy_slice_widths_and_errors(padding=10)
        dx2 = self.dscan.dx**2
        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(widths, errors, pixel_units="m", to_variances=True)
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v

    def tds_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        widths, errors = self.tscan.max_energy_slice_widths_and_errors(padding=10)
        voltages2 = self.tscan.voltage**2

        widths2_m2, errors2_m2 = transform_pixel_widths(widths, errors, pixel_units="m", to_variances=True)
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v

    def beta_scan_fit(self) -> tuple[ValueWithErrorT, ValueWithErrorT]:
        if not self.bscan:
            raise TypeError("Missing optional BetaScan instance.")
        widths, errors = self.bscan.max_energy_slice_widths_and_errors(padding=10)
        beta = self.bscan.beta
        widths2_m2, errors2_m2 = transform_pixel_widths(widths, errors, pixel_units="m", to_variances=True)
        a_beta, b_beta = linear_fit(beta, widths2_m2, errors2_m2)
        return a_beta, b_beta

    def all_fit_parameters(self) -> FittedBeamParameters:
        a_v, b_v = self.tds_scan_fit()
        a_d, b_d = self.dispersion_scan_fit()

        # Values and errors, here we just say there is 0 error in the
        # dispersion and voltage, not strictly true.
        energy = self.tscan.beam_energy(), 0.  # in eV
        dispersion = self.tscan.dx.mean(), 0.
        voltage = self.dscan.voltage[0], 0.

        try:
            a_beta, b_beta = self.beta_scan_fit()
        except TypeError:
            a_beta = b_beta = None

        return FittedBeamParameters(
            a_v=a_v,
            b_v=b_v,
            a_d=a_d,
            b_d=b_d,
            reference_energy=energy,
            reference_dispersion=dispersion,
            reference_voltage=voltage,
            oconfig=self.oconfig,
            a_beta=a_beta,
            b_beta=b_beta
        )


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
        e0j = ufloat(*self.reference_energy) * e # Convert to joules
        k = TDS_WAVENUMBER
        result = (e0j / (dx0 * e * k * v0)) * usqrt(ad - av + dx0**2 * bd)
        return result.n, result.s

    @property
    def sigma_b(self) -> ValueWithErrorT:
        bety = self.oconfig.tds_bety
        alfy = self.oconfig.tds_alfy
        gamy = self.oconfig.tds_gamy
        length = TDS_LENGTH
        sigma_i = ufloat(*self.sigma_i)
        b_beta = sigma_i**2 / (bety + 0.25 * length**2 * gamy - length * alfy)
        result = usqrt(b_beta * self.oconfig.ocr_betx)
        return result.n, result.s

    @property
    def sigma_r(self) -> ValueWithErrorT:
        ad = ufloat(*self.a_d)
        sigma_b = ufloat(*self.sigma_b)
        result = usqrt(ad - sigma_b**2)
        return result.n, result.s

    @property
    def emitx(self) -> ValueWithErrorT:
        gam0 = ufloat(*self.reference_energy) / ELECTRON_MASS_EV
        sigma_b = ufloat(*self.sigma_b)
        result = sigma_b**2 * gam0 / self.oconfig.ocr_betx
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

    def fit_parameters_to_df(self):
        dx0 = self.reference_dispersion
        v0 = self.reference_voltage
        e0 = self.reference_energy

        av, bv = self.a_v, self.b_v
        ad, bd = self.a_d, self.b_d

        pdict = {"V_0": v0,
                 "D_0": dx0,
                 "E_0": e0,
                 "A_V": av,
                 "B_V": bv,
                 "A_D": ad,
                 "B_D": bd}

        if self.a_beta and self.b_beta:
            pdict |= {"A_beta": self.a_beta, "B_beta": self.b_beta}

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _beam_parameters_to_df(self):
        pdict = {"sigma_e": self.sigma_e,
                 "sigma_i": self.sigma_i,
                 "sigma_e_from_tds": self.sigma_e_from_tds,
                 "sigma_b": self.sigma_b,
                 "sigma_r": self.sigma_r,
                 "emitx": self.emitx}

        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"values": values, "errors": errors}, index=pdict.keys())

    def _alt_beam_parameters_to_df(self):
        pdict = {"sigma_e": self.sigma_e_alt,
                 "sigma_i": self.sigma_i_alt,
                 "sigma_e_from_tds": self.sigma_e_from_tds_alt}
        if self.a_beta and self.b_beta:
            pdict |= {"sigma_b": self.sigma_b_alt,
                      "sigma_r": self.sigma_r_alt,
                      "emitx": self.emitx_alt}
        values = []
        errors = []
        for key, pair in pdict.items():
            values.append(pair[0])
            errors.append(pair[1])

        return pd.DataFrame({"alt_values": values, "alt_errors": errors}, index=pdict.keys())

    def beam_parameters_to_df(self):
        params = self._beam_parameters_to_df()
        alt_params = self._alt_beam_parameters_to_df()
        return pd.concat([params, alt_params], axis=1)



def _get_constant_voltage_for_scan(scan):
    # By definition in the dispersion scan the voltages all stay the same.
    # Get dispersion at which the calibration was done
    caldx = scan.calibrator.dispersion
    # Pick
    idx = np.argmin(abs(scan.dx - caldx))
    measurement = scan.measurements[idx]

    metadata = measurement.images[0].metadata
    voltage = scan.calibrator.get_voltage(measurement.tds_percentage,
                                          metadata)
    dx = measurement.dx
    LOG.debug("Deriving constant voltage for dispersion scan.  Calibrator: {scan.calibrator} @ Dx={dx}")
    return np.ones_like(scan.measurements) * voltage
