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
import time
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.constants import c, e, m_e
from scipy.stats import zscore
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt  # pylint: disable=no-name-in-module

from esme.calibration import TDS_LENGTH, TDS_WAVENUMBER
from esme.image import (
    crop_image,
    filter_image,
    get_central_slice_width_from_slice_properties,
    get_cropping_slice,
    get_slice_properties,
)
from esme.maths import ValueWithErrorT, get_gaussian_fit, linear_fit
from esme.optics import calculate_i1d_r34_from_tds_centre

# from esme.plot import formatted_parameter_dfs


PIXEL_SCALE_X_UM: float = 13.7369
PIXEL_SCALE_Y_UM: float = 11.1756


PIXEL_SCALE_X_M: float = PIXEL_SCALE_X_UM * 1e-6
PIXEL_SCALE_Y_M: float = PIXEL_SCALE_Y_UM * 1e-6

LOG = logging.getLogger(__name__)

ELECTRON_MASS_EV: float = m_e * c**2 / e

RawImageT = npt.NDArray


@dataclass
class OpticsFixedPoints:
    beta_screen: float
    beta_tds: float
    alpha_tds: float

    @property
    def gamma_tds(self) -> float:
        return (1 + self.alpha_tds**2) / self.beta_tds


class ImageMetadata:
    def __init__(
        self, kvps: dict[str, Any], optics: OpticsFixedPoints, screen_name: str
    ):
        self.kvps = kvps
        self.optics = optics
        self.screen_name = screen_name
        if "timestamp" not in self.kvps:
            self.timestamp = time.time()
            LOG.warning(
                f"No timestamp in metadata, added arbitrary one: {self.timestamp}"
            )

    @property
    def timestamp(self):
        return self.kvps["timestamp"]

    @timestamp.setter
    def timestamp(self, value):
        self.kvps["timestamp"] = value


class TaggedImage:
    def __init__(
        self,
        image: npt.ArrayLike,
        metadata: ImageMetadata,
        addresses: AnalysisAddresses,
    ):
        self.image = image
        self.metadata = metadata
        self.addresses = addresses

    @property
    def energy(self):
        return self.metadata.kvps[self.addresses.energy]


@dataclass
class SliceAnalysis:
    rows: list
    means: list
    bunch_sigmas: list


class DecomposedBeamImage:
    def __init__(self, pipeline, tagged_raw_image: TaggedImage, crop_all=True):
        self._pipeline = pipeline
        self.tagged_raw_image = tagged_raw_image

        self._image_cache = [tagged_raw_image.image]
        # from IPython import embed; embed()

        self._slice_properties_cache = [
            None
        ]  # No slice properties calculated for raw image initially.

        self._final_cropping_slice = None

        self._crop_all = crop_all
        self._are_cropped = False

    def _check_index_bounds(self, index):
        if index < 0:
            raise ValueError("Cannot have negative image processing step")

        if index > self.pipeline_steps:
            raise ValueError(
                "Requested image processing step is greater than the pipeline length"
            )

    def get_image_at_step(self, index):
        self._check_index_bounds(index)

        try:
            return self._image_cache[index]
        except IndexError:
            pass

        while index >= self._n_cached_images():
            self._do_next_step()

        return self._image_cache[-1]

    def get_slice_properties_at_step(self, index):
        # We assert that is_finished is True because we need the final crop to do any slice analysis.
        if not self.is_finished():
            self.run_to_end()

        slice_properties = self._slice_properties_cache[index]

        if slice_properties is None:
            image_at_step = self.get_image_at_step(index)

            if not self._are_cropped:
                crop = self.final_cropping_slice
                image_at_step = image_at_step[crop]

            slice_properties = get_slice_properties(
                image_at_step, crop=False, fast=False
            )
            row_index, means, sigmas = slice_properties

            everything = np.concatenate(([m.n for m in means], [m.n for m in sigmas]))
            if np.isnan(everything).any():
                raise ValueError()

            self._slice_properties_cache[index] = slice_properties

        return slice_properties

    def run_to_end(self):
        while not self.is_finished():
            self._do_next_step()
        self._crop_all_images()

    def _crop_all_images(self):
        if self._crop_all and not self._are_cropped:
            slc = self.final_cropping_slice
            for i, im in enumerate(self._image_cache):
                self._image_cache[i] = im[
                    slc
                ].copy()  # Copy to force GC of original array.
        self._are_cropped = True

    def is_finished(self):
        return self.pipeline_steps + 1 == len(self._image_cache)

    def pipeline_attributes(self, attrname: str):
        return [gettatr(fn, attrname) in self._pipeline]

    def _n_cached_images(self) -> int:
        return len(self._image_cache)

    @property
    def pipeline_steps(self) -> int:
        return len(self._pipeline.functors)

    @property
    def im0(self) -> np.ndarray:
        return self._image_cache[0]

    @cached_property
    def apparent_bunch_lengths(self):
        pass

    @cached_property
    def slice_analysis(self) -> SliceAnalysis:
        # It has to be cropped, otherwise the algorithm doesn't work anyway.
        self.final_image
        if not self._are_cropped:
            cslice = self.final_cropping_slice
        return image.get_slice_properties(image[cslice], fast=False)

    def _do_next_step(self):
        im0 = self._image_cache[-1]
        start_index = len(self._image_cache) - 1
        next_functor = self._pipeline.functors[start_index]
        im1 = next_functor(im0)
        self._image_cache.append(im1)
        self._slice_properties_cache.append(None)

    def final_image(self):
        if not self.is_finished():
            self.run_to_end()
        return self._image_cache[-1]

    def final_slice_analysis(self):
        return self.get_slice_properties_at_step(self.pipeline_steps)

    @cached_property
    def final_cropping_slice(self):
        return get_cropping_slice(self.final_image())

    def max_energy_row_index(self):
        rows1, means1, sigmas1 = self.final_slice_analysis()
        return means1.argmin()


@dataclass
class AnalysisAddresses:
    image: str
    energy: str
    amplitude_sp: str
    amplitude_rb: str
    power_sp: str


@dataclass
class ScanMetadata:
    """Minimal set of addresses that are necessary for the analysis and measurement"""

    addresses: Addresses
    fixed_optics: OpticsFixedPoints


class SliceWidths:
    def __init__(self, x):
        self.x = x


class ScanType(Enum):
    DISPERSION = "DISPERSION"
    TDS = "TDS"
    BETA = "BETA"

    @classmethod
    @property
    def ALT_NAME_MAP(cls):
        return {cls.DISPERSION: "dscan", cls.TDS: "tscan", cls.BETA: "bscan"}

    @classmethod
    def from_string(cls, string):
        """Accepts either ScanType.TDS or TDS, same for DISPERSION and BETA"""
        return cls(string.removeprefix("ScanType."))

    def alt_name(self):
        return self.ALT_NAME_MAP[self]


class Scan:
    def __init__(self, setpointdfs):
        self.setpointdfs = setpointdfs

    def voltages(self):
        return np.array([sdf.voltage for sdf in self.setpointdfs])

    def dispersions(self):
        return np.array([sdf.dispersion for sdf in self.setpointdfs])

    def betas(self):
        return np.array([sdf.beta for sdf in self.setpointdfs])

    def amplitude_rbs(self):
        return np.array([sdf.amplitude_rbs for sdf in self.setpointdfs])

    def amplitude_sps(self):
        return np.array([sdf.amplitude_sp for sdf in self.setpointdfs])

    def beam_energies(self):
        return np.array([sdf.beam_energies() for sdf in self.setpointdfs])

    def sortby(self, attr):
        self.setpointdfs.sort(key=lambda x: getattr(x, attr))
        return self

    def __iter__(self):
        yield from iter(self.setpointdfs)


class MeasurementDataFrames:
    def __init__(self, *, dscans, tscans, bscans, optics, bg=None):
        # Copy and sort by each of their respective independent variables
        self.dscan = dscans.sortby("dispersion")
        self.tscan = tscans.sortby("voltage")
        self.bscan = bscans.sortby("beta")
        self.optics = optics
        self.bg = bg

    def nfullscans(self) -> int:
        return bool(self.dscan) + bool(self.tscan) + bool(self.bscan)

    def energy(self):
        energies = []
        energies.extend([t.energy for t in self.tscan])
        energies.extend([b.energy for b in self.bscan])
        energies.extend([d.energy for d in self.dscan])
        return np.mean(energies)

    def max_voltage_df(self):
        index = np.argmax(self.tscan.voltages())
        max_voltage_df = self.tscan.setpointdfs[index]
        return max_voltage_df

    def tscan_dispersion(self):
        return self.tscan.dispersions()[0]

    def dscan_voltage(self):
        return self.dscan.voltages()[0]


class SetpointDataFrame:
    def __init__(self, df, addresses, optics, bg=0.0):
        self.df = df
        self.addresses = addresses
        self.optics = optics
        self.bg = bg

    @property
    def image_address(self):
        return self.addresses.image

    @property
    def energy_address(self):
        return self.addresses.energy

    @property
    def image_paths(self):
        yield from iter(self.df[self.image_address])

    def get_tagged_images(self):
        screen_name = self.screen_name
        for _, dfrow in self.df.iterrows():
            path = dfrow[self.image_address]
            image = np.load(path)["image"]
            image = image.T
            metadata = ImageMetadata(dfrow, self.optics, screen_name)
            yield TaggedImage(image, metadata, self.addresses)

    def get_images_with_background(self):
        for path in self.image_paths:
            image = np.load(path)["image"]
            image = image.T  # Flip to match control room..?  TODO
            yield image, self.bg

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
    def amplitude_rbs(self):
        return np.array(self.df[self.addresses.amplitude_rb])

    @property
    def amplitude_sp(self):
        return _get_constant_column_from_df(self.df, self.addresses.amplitude_sp)

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

    @property
    def name(self):
        scan_str = ScanType.from_string(self.scan_type).alt_name()
        return f"{scan_str}:V={self.voltage/1e6:.2f}MV, D={self.dispersion}m, β={self.beta}m"

    def beam_energies(self):
        return np.array(self.df[self.addresses.energy])


def _get_constant_column_from_df(df, name):
    col = df[name]
    unique_values = set(col)
    if len(unique_values) != 1:
        raise ValueError(f"Column {name} is not constant in dataframe")
    return unique_values.pop()


def make_outlier_widths_mask(widths, sigma_cut):
    # Returns mask where True means the entry IS an outlier.
    assert sigma_cut >= 0
    # We do thsi repeatedly until we remove all outliers.  We have to
    # do it multiple times in case the presence of one extreme outlier
    # masks the presence of a less extreme outlier (that is
    # nevertheless still an outlier...).

    # Start with masked array with all values valid
    widths = np.ma.array(widths, mask=np.zeros_like(widths))

    while True:
        z = zscore(widths, ddof=1)
        maski = abs(z) >= sigma_cut

        if maski.sum() == 0:  # If no outliers (left) then we are done.
            break

        widths.mask |= maski  # Else binary or with mask, building it up bit by bit...

    # Have to negate because 0 = not mask and 1 = masked in masked
    # arrays, but we want the usual style, which is the opposite.
    return widths.mask


def count_outliers(x):
    return sum(abs(z) >= abs(sigma_cut))


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
    energy = measurement.energy * 1e6  # in eV

    width_pixels = measurement.mean_central_slice_width_with_error(padding=10)
    # Calculate with uncertainties automatically.
    width_pixels_unc = ufloat(*width_pixels)
    width_unc = width_pixels_unc * PIXEL_SCALE_X_M

    energy_spread_unc = energy * width_unc / dx
    energy_spread_ev = energy_spread_unc

    value, error = energy_spread_ev.n, energy_spread_ev.s

    return value, error  # in eV


def process_measurment_dfs(measurement, avmapping):
    tscan_widths = {}
    for tscan_setpoint in measurement.tscan:
        tscan_widths[tscan_setpoint.voltage] = pixel_widths_from_setpoint(
            tscan_setpoint
        )

    dscan_widths = {}
    for dscan_setpoint in measurement.dscan:
        dscan_widths[dscan_setpoint.dispersion] = pixel_widths_from_setpoint(
            dscan_setpoint
        )

    bscan_widths = {}
    for bscan_setpoint in measurement.bscan:
        bscan_widths[dscan_setpoint.beta] = pixel_widths_from_setpoint(bscan_setpoint)

    # in metres
    # bunch_length = calculate_bunch_length(measurement.max_voltage_df())
    # sigma_z_rms = calculate_bunch_length(measurement.max_voltage_df(), length="rms")

    np.argmax(measurement.tscan.voltages())
    max_voltage_df = measurement.max_voltage_df()

    bunch_length = true_bunch_length_from_df(max_voltage_df, method="gaussian")
    sigma_z_rms = true_bunch_length_from_df(max_voltage_df, method="rms")

    ofp = measurement.optics

    fitter = SliceWidthsFitter(dscan_widths, tscan_widths)
    params = fitter.all_fit_parameters(
        measurement.energy() * 1e6,  # to eV
        dscan_voltage=measurement.dscan_voltage(),
        tscan_dispersion=measurement.tscan_dispersion(),
        optics_fixed_points=ofp,
        sigma_z=(bunch_length.n, bunch_length.s),
        sigma_z_rms=(sigma_z_rms.n, sigma_z_rms.s),
    )

    beam_df = params.beam_parameters_to_df()
    fit_df = params.fit_parameters_to_df()

    # If failed to reconstruct values...
    if np.isnan(beam_df.loc["sigma_i"]["values"]):
        sigma_e = _simple_calc(measurement.max_dispersion_sp())
        beam_df.loc["sigma_e"] = {
            "values": sigma_e.n,
            "errors": sigma_e.s,
            "alt_values": np.nan,
            "alt_errors": np.nan,
        }

    dirname = "result"

    beam_df.to_pickle(f"{dirname}-beam.pkl")
    fit_df.to_pickle(f"{dirname}-fit.pkl")

    fit_df, beam_df = formatted_parameter_dfs(params)

    return fit_df, beam_df


def pixel_widths_from_setpoint(setpoint: SetpointDataFrame, policy="emax"):
    pass

    central_sigmas = []
    for image, bg in setpoint.get_images_with_background():
        image = filter_image(image, bg=bg, crop=True)

        _, means, bunch_sigmas = get_slice_properties(image, fast=True)
        # [x.n for x in bunch_sigmas]
        # if np.isnan(bunch_sigmas).any():
        #     import ipdb; ipdb.set_trace()

        sigma = get_central_slice_width_from_slice_properties(
            means, bunch_sigmas, padding=5
        )

        try:
            if np.isnan(sigma):
                continue
        except:
            # coontinue
            pass

        # if policy == "emax":
        #     central_width_row = np.argmin(means)
        # elif policy == "middle":
        #     print("middle!")
        #     central_width_row = len(means) // 2
        # else:
        #     raise ValueError(f"Unknown policy: {policy}")

        central_sigmas.append(sigma)

    return np.mean(central_sigmas)


# def full_beam_slice_widths


class SliceWidthsFitter:
    def __init__(self, dscan_widths, tscan_widths, bscan_widths=None, *, avmapping):
        self.dscan_widths = dscan_widths
        self.tscan_widths = tscan_widths
        self.bscan_widths = bscan_widths
        self.avmapping = avmapping

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
        amplitudes = np.array(list(self.tscan_widths.keys()))
        try:
            voltages = self.avmapping(amplitudes)
        except TypeError:
            raise TypeError()

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

    def all_fit_parameters(
        self,
        beam_energy,
        tscan_dispersion,
        dscan_voltage,
        optics_fixed_points,
        sigma_z=None,
        sigma_z_rms=None,
    ) -> DerivedBeamParameters:
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

        return DerivedBeamParameters(
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
            sigma_z=sigma_z,
            sigma_z_rms=sigma_z_rms,
        )


@dataclass
class DerivedBeamParameters:
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
    a_beta: Optional[ValueWithErrorT] = None
    b_beta: Optiona[ValueWithErrorT] = None
    sigma_z: Optional[ValueWithErrorT] = None
    sigma_z_rms: Optional[ValueWithErrorT] = None

    def a_d_derived(self, sigma_r, emittance: Optional[float] = None):
        if emittance is None:
            emittance = self.emitx
        return np.sqrt(sigma_r**2 + (emittance * self.oconfig.beta_screen) ** 2)

    def set_a_d_to_known_value(self, sigma_r, emittance: Optional[float] = None):
        new_ad = self.a_d_derived(sigma_r, emittance=emittance)
        self.a_d = new_ad

    def set_a_v_to_known_value(self, sigma_r, emittance: Optional[float] = None):
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

    def beam_parameters_to_df(self) -> pd.DataFrame:
        # sigma_t in picosecondsm
        params = self._beam_parameters_to_df()
        alt_params = self._alt_beam_parameters_to_df()
        if self.sigma_z is not None:
            self.sigma_t
            params.loc["sigma_z"] = {
                "values": self.sigma_z[0],
                "errors": self.sigma_z[1],
            }
            params.loc["sigma_t"] = {
                "values": self.sigma_t[0],
                "errors": self.sigma_t[1],
            }

        return pd.concat([params, alt_params], axis=1)

    @property
    def sigma_t(self):
        stn = self.sigma_z[0] / c
        ste = self.sigma_z[1] / c
        return stn, ste

    @property
    def sigma_t_rms(self):
        stn = self.sigma_z_rms[0] / c
        ste = self.sigma_z_rms[1] / c
        return stn, ste


def true_bunch_length_from_setpoint(
    setpoint, avmapping: AmplitudeVoltageMapping, method: str = "gaussian"
):
    apparent_bunch_lengths = []
    if method == "gaussian":
        fn = apparent_gaussian_bunch_length_from_processed_image
    elif method == "rms":
        fn = apparent_rms_bunch_length_from_processed_image
    else:
        raise ValueError(f"Unknown method: {method}")

    for dbim in setpoint.images:
        im = dbim.final_image()
        df = dbim.tagged_raw_image.metadata
        length, error = fn(im)
        apparent_bunch_lengths.append(ufloat(length, error))

    mean_apparent_bunch_length = np.mean(apparent_bunch_lengths)

    screen_name = setpoint.screen_name

    energies_mev = setpoint.energies()
    energy_mev = energies_mev.mean()
    energy_joules = energy_mev * 1e6 * e

    r34 = calculate_i1d_r34_from_tds_centre(setpoint.df, screen_name, energy_mev)

    tds_amplitude = setpoint.amplitude
    voltage = avmapping(tds_amplitude)
    bunch_length = (
        (energy_joules / (e * voltage * TDS_WAVENUMBER))
        * mean_apparent_bunch_length
        / r34
    )
    return abs(bunch_length)


def true_bunch_length_from_df(setpoint, method: str = "gaussian"):
    from IPython import embed

    embed()
    apparent_bunch_lengths = []
    if method == "gaussian":
        fn = apparent_gaussian_bunch_length_from_processed_image
    elif method == "rms":
        fn = apparent_rms_bunch_length_from_processed_image
    else:
        raise ValueError(f"Unknown method: {method}")

    for raw_im, bg in setpoint.get_images_with_background():
        # image = np.load(path)["image"]
        # image = image.T # Flip to match control room..?  TODO
        # XXX: I do not use any bg here...
        image = filter_image(raw_im, bg=bg, crop=True)
        length, error = fn(image)
        apparent_bunch_lengths.append(ufloat(length, error))

    mean_apparent_bunch_length = np.mean(apparent_bunch_lengths)

    r34 = abs(
        calculate_i1d_r34_from_tds_centre(
            setpoint.df, setpoint.screen_name, setpoint.energy
        )
    )
    energy = setpoint.energy * 1e6 * e  # to Joules
    voltage = setpoint.voltage

    bunch_length = (
        (energy / (e * voltage * TDS_WAVENUMBER)) * mean_apparent_bunch_length / r34
    )

    return bunch_length


# def true_bunch_length_from_imagesf(processed_images, df, method="gaussian"):
#     apparent_bunch_lengths = []
#     if method == "gaussian":
#         fn = apparent_gaussian_bunch_length_from_processed_image
#     elif method == "rms":
#         fn = apparent_rms_bunch_length_from_processed_image
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     for im in processed_images:
#         length, error = fn(image)
#         apparent_bunch_lengths.append(ufloat(length, error))

#     mean_apparent_bunch_length = np.mean(apparent_bunch_lengths)

#     r34 = abs(calculate_i1d_r34_from_tds_centre(setpoint.df, setpoint.screen_name, setpoint.energy))
#     energy = setpoint.energy * 1e6 * e # to Joules
#     voltage = setpoint.voltage
#     bunch_length = (energy / (e * voltage * TDS_WAVENUMBER)) * mean_apparent_bunch_length / r34

#     return bunch_length


def true_bunch_length_from_df(setpoint, method="gaussian"):
    apparent_bunch_lengths = []
    if method == "gaussian":
        fn = apparent_gaussian_bunch_length_from_processed_image
    elif method == "rms":
        fn = apparent_rms_bunch_length_from_processed_image
    else:
        raise ValueError(f"Unknown method: {method}")

    for raw_im, bg in setpoint.get_images_with_background():
        # image = np.load(path)["image"]
        # image = image.T # Flip to match control room..?  TODO
        # XXX: I do not use any bg here...
        image = filter_image(raw_im, bg=bg, crop=True)
        length, error = fn(image)
        apparent_bunch_lengths.append(ufloat(length, error))

    mean_apparent_bunch_length = np.mean(apparent_bunch_lengths)

    r34 = abs(
        calculate_i1d_r34_from_tds_centre(
            setpoint.df, setpoint.screen_name, setpoint.energy
        )
    )
    energy = setpoint.energy * 1e6 * e  # to Joules
    voltage = setpoint.voltage
    bunch_length = (
        (energy / (e * voltage * TDS_WAVENUMBER)) * mean_apparent_bunch_length / r34
    )

    return bunch_length


def streaking_parameter(*, voltage, energy, r12_streaking, wavenumber=TDS_WAVENUMBER):
    """Energy in eV.  Voltage in V."""
    energy = energy * e  # in eV and convert to joules
    k0 = e * voltage * wavenumber / energy
    streak = r12_streaking * k0
    return streak


def apparent_gaussian_bunch_length_from_processed_image(image):
    image = crop_image(image)
    pixel_indices = np.arange(image.shape[0])  # Assumes streaking is in image Y
    projection = image.sum(axis=1)
    try:
        popt, perr = get_gaussian_fit(pixel_indices, projection)
    except RuntimeError:
        # This can happen when for example when the cropping fails completely...
        return np.nan, np.nan

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


def apparent_rms_bunch_length_from_processed_image(image):
    image = crop_image(image)
    pixel_indices = np.arange(image.shape[0])  # Assumes streaking is in image Y
    # Sum onto streaking axis to get longitudinal property
    projection = image.sum(axis=1)

    # Get PDF of longitudinal distribution by normalising w.r.t
    beam_pdf = projection / projection.sum()
    population_mean = np.trapz(beam_pdf * pixel_indices, x=pixel_indices)
    population_variance = np.trapz(
        beam_pdf * (pixel_indices - population_mean) ** 2, x=pixel_indices
    )
    population_standard_deviation = np.sqrt(population_variance)

    # Transform units from px to whatever was chosen
    mean_length, mean_error = transform_pixel_widths(
        [population_standard_deviation],
        [0],
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
