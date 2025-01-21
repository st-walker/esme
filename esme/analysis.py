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
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.constants import c, e, m_e
from scipy.stats import zscore
from uncertainties import ufloat
from uncertainties.umath import sqrt as usqrt  # pylint: disable=no-name-in-module

from esme.calibration import TDS_WAVENUMBER
from esme.image import (
    crop_image,
    filter_image,
    get_central_slice_width_from_slice_properties,
    get_cropping_slice,
    get_slice_properties,
)
from esme.maths import get_gaussian_fit
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
        [0],
        to_variances=False,
        pixel_units="m",
        dimension="y",
    )

    return np.squeeze(mean_length), np.squeeze(mean_error)


def true_bunch_length_from_processed_image(image, *, voltage, r34, energy) -> ufloat:
    raw_bl, raw_bl_err = apparent_gaussian_bunch_length_from_processed_image(image)
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
