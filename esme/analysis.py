"""Units: everything is in eV, volts, etc."""

from __future__ import annotations

import contextlib
import logging
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.ndimage as ndi
from scipy.constants import c, e, m_e
from scipy.optimize import curve_fit
from uncertainties import ufloat, umath

from esme.calibration import TDS_WAVENUMBER, TDS_LENGTH


IMAGE_PATH_KEY: str = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"


NOISE_THRESHOLD: float = 0.08  # By eye...

PIXEL_SCALE_X_UM: float = 13.7369
PIXEL_SCALE_Y_UM: float = 11.1756

PIXEL_SCALE_X_M: float = PIXEL_SCALE_X_UM * 1e-6
PIXEL_SCALE_Y_M: float = PIXEL_SCALE_Y_UM * 1e-6

LOG = logging.getLogger(__name__)

ELECTRON_MASS_EV: float = m_e * c**2 / e

RawImageT = npt.NDArray
ValueWithErrorT = tuple[float, float]

MULTIPROCESSING = True

CENTRAL_SLICE_SEARCH_WINDOW_RELATIVE_WIDTH = 9


def line(x, a0, a1) -> Any:
    return a0 + a1 * x


def get_slice_properties(image: RawImageT, fast=True) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    #  Get bounds of image (i.e. to remove all fully-zero rows and columns)---this
    # speeds up the fitting procedure a lot by only fitting region of interest.
    (irow0, irow1), (icol0, icol1) = get_cropping_bounds(image, just_central_slices=fast)

    # Do the actual cropping
    imcropped = image[irow0:irow1, icol0:icol1]

    row_index = np.arange(imcropped.shape[1])

    mean_slice_position = []
    mean_slice_position_error = []
    sigma_slice = []
    sigma_slice_error = []
    for beam_slice in imcropped: # Iterates over the ROWS, so each one is a slice of the beam.
        try:
            popt, perr = get_gaussian_fit(row_index, beam_slice)
        except RuntimeError:  # Happens if curve_fit fails to converge.
            # Set parameters to NaN, mask them later from the output
            mu = sigma = sigma_mu = sigma_sigma = np.nan
        else:
            _, mu, sigma = popt
            _, sigma_mu, sigma_sigma = perr

        mean_slice_position.append(mu)
        mean_slice_position_error.append(sigma_mu)
        sigma_slice.append(sigma)
        sigma_slice_error.append(sigma_sigma)

    # So we get back into the coordinate system of the original, uncropped image:
    row_index = np.arange(irow0, irow1)
    mean_slice_position += icol0

    # Deal with nans due to for example
    nan_mask = ~(np.isnan(mean_slice_position) | np.isnan(mean_slice_position_error) | np.isnan(sigma_slice) | np.isnan(sigma_slice_error))

    mean_slice_position = np.array([ufloat(n, s) for n, s in zip(mean_slice_position, mean_slice_position_error)])
    slice_width = np.array([ufloat(n, s) for n, s in zip(sigma_slice, sigma_slice_error)])

    row_index = row_index[nan_mask]
    mean_slice_position = mean_slice_position[nan_mask]
    slice_width = slice_width[nan_mask]

    return row_index, mean_slice_position, slice_width


def gauss(x, a, mu, sigma) -> Any:
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def get_cropping_bounds(im: RawImageT, just_central_slices=False) -> tuple[tuple[int, int], tuple[int, int]]:
    non_zero_mask = im != 0

    # "Along axis 1" -> each input to np.any is a row (axis 1 "points to the
    # right"), so gives indices for axis 0, i.e. rows!
    non_zero_row_indices = np.squeeze(np.where(non_zero_mask.any(axis=1)))
    # "Along axis 0" -> each input to np.any is a column (axis 0 "points down"
    # (im[0] gives a row of pixels for example, not a column)), so gives indices
    # for axis 1, i.e. columns!
    non_zero_column_indices = np.squeeze(np.where(non_zero_mask.any(axis=0)))

    irow0 = non_zero_row_indices[0]
    irow1 = non_zero_row_indices[-1]
    icol0 = non_zero_column_indices[0]
    icol1 = non_zero_column_indices[-1]

    if just_central_slices:
        length = (irow1 - irow0)
        middle = irow0 + length // 2
        irow0 = middle - length // CENTRAL_SLICE_SEARCH_WINDOW_RELATIVE_WIDTH
        irow1 = middle + length // CENTRAL_SLICE_SEARCH_WINDOW_RELATIVE_WIDTH

    # Add 1 as index is exlusive on the upper bound, and this
    # sometimes matters and prevents bugs/crashes in the error
    # calculation later, because we can end up with empty rows or
    # columns.
    return (irow0, irow1 + 1), (icol0, icol1 + 1)


def get_cropping_slice(im: RawImageT) -> tuple:
    (row0, row1), (col0, col1) = get_cropping_bounds(im)
    return np.s_[row0, row1:col0, col1]


def crop_image(im: RawImageT) -> RawImageT:
    (idx_row0, idx_row1), (idx_col0, idx_col1) = get_cropping_bounds(im)
    return im[idx_row0:idx_row1, idx_col0:idx_col1]


def process_image(im0: RawImageT, bg: RawImageT) -> RawImageT:
    # Subtract bg from image.
    im = im0 - bg

    # Set negative due to bg subtraction to zero.
    im0bg = im.clip(min=0)

    # Apply uniform filter to try and smear out the big, isolated (background)
    # values
    im0bgu = ndi.uniform_filter(im0bg, size=100)

    # Get mask for all pixels which, when smeared out, are below some max value
    # of the image. This should result in very isolated pixels getting set to 0,
    # and leave core beam pixels untouched, because they have many neighbours.
    mask = im0bgu < NOISE_THRESHOLD * im0bgu.max()

    # img2 = ndi.median_filter(im0bg, size=10)
    img1 = ndi.uniform_filter(im0bg, size=3)
    inds_hi = (1.5 * img1) < im0bg

    # Apply mask to original bg-subtracted image.
    im0bg[mask | inds_hi] = 0

    im_no_outliers = remove_all_disconnected_pixels(im0bg)

    return im_no_outliers


def remove_all_disconnected_pixels(im: RawImageT) -> RawImageT:
    # normalize. usine 16bit usigned int because that's what the original raw
    # image (pcl) files come as. Keep lower and upper bounds the same as the
    # original image so that in principle different processed images are perhaps
    # comparable.
    imu8 = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    _, thresh = cv2.threshold(imu8, 0, 255, cv2.THRESH_BINARY)
    _, markers = cv2.connectedComponents(thresh)
    ranked_markers, counts = np.unique(markers.flatten(), return_counts=True)
    # marker corresponds to the background, the second one corresponds to the
    # main contiguous space occupied by the beam
    beam_blob_marker = np.argpartition(-counts, kth=1)[1]
    mask = markers == beam_blob_marker
    masked_image = np.where(mask, im, 0)

    return masked_image


def get_slice_core(pixels) -> tuple[npt.NDArray, npt.NDArray]:
    # Remove zeroes on either side of the slice and just get the
    # values where there is signal.
    nonzero_pixels = (pixels != 0).nonzero()[0]
    istart = nonzero_pixels.min()
    iend = nonzero_pixels.max()

    pixelcut = pixels[istart : iend + 1]
    pixel_index = np.arange(len(pixelcut))

    return pixel_index, pixelcut


def get_gaussian_fit(x, y) -> tuple[tuple, tuple]:
    """popt/perr order: a, mu, sigma"""
    mu0 = y.argmax()
    a0 = y.max()
    sigma0 = 1

    # Bounds argument of curve_fit slows the fitting procedure down too much
    # (>2x worse), so avoid using it here.
    popt, pcov = curve_fit(
        gauss,
        x,
        y,
        p0=[a0, mu0, sigma0],
    )
    variances = np.diag(pcov)
    if (variances < 0).any():
        raise RuntimeError(f"Negative variance detected: {variances}")
    perr = np.sqrt(variances)
    return popt, perr


class TDSScreenImage:
    def __init__(self, metadata):
        self.metadata = metadata
        self._image = None

    @property
    def filename(self) -> str:
        return self.metadata[IMAGE_PATH_KEY]

    def to_im(self, process=True) -> RawImageT:
        # path to png is in the df, but actuallt we want path to the adjacent
        # pcl file.
        fname = Path(self.filename).with_suffix(".pcl")
        im = pickle.load(open(fname, "rb"))
        # Flip to match what we see in the control room.  not sure if
        # I need an additional flip here or not, but shouldn't matter too much.
        return im.T

    def show(self) -> None:
        im = self.to_im()
        fig, ax = plt.subplots()
        ax.imdraw(im)
        plt.show()

    @property
    def is_bg(self) -> bool:
        return not bool(self.metadata["XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"])

    @property
    def beam_energy(self) -> float:
        eV = 1e6  # Energy is given in MeV which we convert to eV for consistency.
        return self.metadata["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"] * eV


class ScanMeasurement:
    DF_DX_SCREEN_KEY = "MY_SCREEN_DX"
    DF_BETA_SCREEN_KEY = "MY_SCREEN_BETA"
    DF_TDS_PERCENTAGE_KEY = "MY_TDS_AMPL"

    def __init__(self, df_path: os.PathLike, # calibrator=None,
                 bad_image_indices=None):
        """bad_image_indices are SKIPPED and not loaded at all."""
        LOG.debug(f"Loading measurement: {df_path=} with {bad_image_indices=}")
        df_path = Path(df_path)


        df = pd.read_pickle(df_path)

        self.dx = _get_constant_key_from_df_safeley(df, self.DF_DX_SCREEN_KEY)
        self.tds_percentage = _get_constant_key_from_df_safeley(df, self.DF_TDS_PERCENTAGE_KEY)

        try:
            self.beta = _get_constant_key_from_df_safeley(df, self.DF_BETA_SCREEN_KEY)
        except KeyError:
            pass

        self.images = []
        if bad_image_indices is None:
            bad_image_indices = []
        self.bg = []
        self._mean_bg_im: Optional[RawImageT] = None  # For caching.

        for i, relative_path in enumerate(df[IMAGE_PATH_KEY]):
            abs_path = df_path.parent / relative_path
            LOG.debug(f"Loading image index {i} @ {relative_path}")
            if i in bad_image_indices:
                LOG.debug(f"Skipping bad image: {i}")
                continue
            # In the df the png paths relative to the pickled dataframe, but
            # I want to be able to call it from anywhere, so resolve them to
            # absolute paths.
            abs_path = df_path.parent / relative_path
            metadata = df[df[IMAGE_PATH_KEY] == relative_path].squeeze()
            metadata[IMAGE_PATH_KEY] = abs_path
            image = TDSScreenImage(metadata)
            if image.is_bg:
                LOG.debug(f"Image{i} is bg")
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
        bad_images_per_measurement=None,
    ):
        # Ideally this voltage, calibration etc. stuff should go in the df.
        # for now, whatever.
        LOG.debug(f"Instantiating {type(self).__name__}")
        if bad_images_per_measurement is None:
            bad_images_per_measurement = len(files) * [None]

        self.calibrator = calibrator

        self.measurements = []
        for i, df_path in enumerate(files):
            measurement = ScanMeasurement(
                df_path,
                # calibrator=calibrator,
                bad_image_indices=bad_images_per_measurement[i],
            )
            self.measurements.append(measurement)

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


def linear_fit(indep_var, dep_var, dep_var_err) -> tuple[ValueWithErrorT, ValueWithErrorT]:
    popt, pcov = curve_fit(line, indep_var, dep_var, sigma=dep_var_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0]
    a1 = popt[1], perr[1]

    return a0, a1


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
        result = (energy0 / dx0) * umath.sqrt(av - ad)
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
        result = (energy0 / dx0) * umath.sqrt(bd * dx0**2 - bv * v0**2)
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
            result = (e0_joules / (dx0 * e * k)) * umath.sqrt(bv)
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
        result = (e0j / (dx0 * e * k * v0)) * umath.sqrt(ad - av + dx0**2 * bd)
        return result.n, result.s

    @property
    def sigma_b(self) -> ValueWithErrorT:
        bety = self.oconfig.tds_bety
        alfy = self.oconfig.tds_alfy
        gamy = self.oconfig.tds_gamy
        length = TDS_LENGTH
        sigma_i = ufloat(*self.sigma_i)
        b_beta = sigma_i**2 / (bety + 0.25 * length**2 * gamy - length * alfy)
        result = umath.sqrt(b_beta * self.oconfig.ocr_betx)
        return result.n, result.s

    @property
    def sigma_r(self) -> ValueWithErrorT:
        ad = ufloat(*self.a_d)
        sigma_b = ufloat(*self.sigma_b)
        result = umath.sqrt(ad - sigma_b**2)
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
        result = umath.sqrt(ad + bd * d0**2 - ab)
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
        result = umath.sqrt(ab - bd * d0**2)
        return result.n, result.s

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
                 "sigma_i": self.sigma_i_alt}
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
