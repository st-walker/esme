"""Units: everything is in eV, volts, etc."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable
import functools

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
from scipy.constants import c, e, m_e
from scipy.optimize import curve_fit
from uncertainties import ufloat, umath


IMAGE_PATH_KEY = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"


NOISE_THRESHOLD = 0.08  # By eye...

# WHICH DO I USE??

PIXEL_SCALE_X_UM = 13.7369
PIXEL_SCALE_Y_UM = 11.1756

PIXEL_SCALE_X_M = PIXEL_SCALE_X_UM * 1e-6
PIXEL_SCALE_Y_M = PIXEL_SCALE_Y_UM * 1e-6

LOG = logging.getLogger(__name__)

ELECTRON_MASS_EV = m_e * c**2 / e

RawImageT = npt.NDArray


def get_slice_properties(image: RawImageT) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    #  Get bounds of image (i.e. to remove all fully-zero rows and columns)---this
    # speeds up the fitting procedure a lot by only fitting region of interest.
    (irow0, irow1), (icol0, icol1) = get_cropping_bounds(image)

    # Do the actual cropping
    imcropped = image[irow0:irow1, icol0:icol1]

    columns = imcropped.T
    row_index = np.arange(columns.shape[1])

    means = []
    mean_sigmas = []
    sigmas = []
    sigma_sigmas = []
    for i, column in enumerate(columns):
        try:
            popt, perr = get_gaussian_fit(row_index, column)
        except RuntimeError:  # Happens if curve_fit fails to converge.
            # Set parameters to NaN, mask them later from the output
            mu = sigma = sigma_mu = sigma_sigma = np.nan
        else:
            _, mu, sigma = popt
            _, sigma_mu, sigma_sigma = perr

        means.append(mu)
        mean_sigmas.append(sigma_mu)
        sigmas.append(sigma)
        sigma_sigmas.append(sigma_sigma)

    # So we get back into the coordinate system of the original, uncropped image:
    column_index = np.arange(icol0, icol1)
    means += irow0

    # Deal with nans due to for example
    nan_mask = ~(np.isnan(means) | np.isnan(mean_sigmas) | np.isnan(sigmas) | np.isnan(sigma_sigmas))

    means = np.array([ufloat(n, s) for n, s in zip(means, mean_sigmas)])
    sigmas = np.array([ufloat(n, s) for n, s in zip(sigmas, sigma_sigmas)])

    column_index = column_index[nan_mask]
    means = means[nan_mask]
    sigmas = sigmas[nan_mask]

    return column_index, means, sigmas


def gauss(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def line(x, a0, a1):
    return a0 + a1 * x


def get_cropping_bounds(im: RawImageT, image_index=-1) -> tuple[tuple[int, int], tuple[int, int]]:
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

    # Add 1 as index is exlusive on the upper bound, and this sometimes matters
    # and prevents bugs/crashes in the error calculation later, because we can
    # up with empty rows or columns.
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


def get_slice_core(pixels):
    nonzero_pixels = (pixels != 0).nonzero()[0]
    istart = nonzero_pixels.min()
    iend = nonzero_pixels.max()

    pixelcut = pixels[istart : iend + 1]
    pixel_index = np.arange(len(pixelcut))

    return pixel_index, pixelcut


def _dispersion_from_filename(fname: os.PathLike) -> float:
    path = Path(fname)
    match = re.search(r"Dx_[0-9]+", path.stem)

    if not match:
        raise MissingMetadataInFileNameError(fname)
    substring = match.group(0)
    substring.split("Dx_")

    dx = float(match.group(0).split("Dx_")[1])

    return dx / 1000  # convert to metres


def _tds_magic_number_from_filename(fname: os.PathLike) -> int:
    path = Path(fname)
    match = re.search(r"tds_[0-9]+", path.stem)

    if not match:
        raise MissingMetadataInFileNameError(fname)
    substring = match.group(0)
    substring.split("tds_")

    tds_magic_number = int(match.group(0).split("tds_")[1])

    return tds_magic_number


def get_gaussian_fit(x, y):
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


class MissingMetadataInFileNameError(RuntimeError):
    pass


class TDSScreenImage:
    def __init__(self, metadata):
        self.metadata = metadata
        self._image = None

    @property
    def filename(self):
        return self.metadata[IMAGE_PATH_KEY]

    def to_im(self, process=True):
        # path to png is in the df, but actuallt we want path to the adjacent
        # pcl file.
        fname = Path(self.filename).with_suffix(".pcl")
        im = pickle.load(open(fname, "rb"))
        return im

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
        eV = 1e6 # Energy is given in MeV which we convert to eV for consistency.
        return self.metadata["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"] * eV


class ScanMeasurement:
    def __init__(self, df_path: os.PathLike):
        df_path = Path(df_path)
        self.dx = _dispersion_from_filename(df_path)
        self.tds = _tds_magic_number_from_filename(df_path)
        self.images = []
        self.bg = []
        self._mean_bg_im = None  # For caching.
        with df_path.open("br") as f:
            df = pickle.load(f)
            for relative_path in df[IMAGE_PATH_KEY]:
                # In the df the png paths relative to the pickled dataframe, but
                # I want to be able to call it from anywhere, so resolve them to
                # absolute paths.
                abs_path = df_path.parent / relative_path
                metadata = df[df[IMAGE_PATH_KEY] == relative_path].squeeze()
                metadata[IMAGE_PATH_KEY] = abs_path
                image = TDSScreenImage(metadata)
                if image.is_bg:
                    self.bg.append(image)
                elif not image.is_bg:
                    self.images.append(image)


    def __repr__(self) -> str:
        return f"<Measurement: Dx={self.dx}>"

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

    def mean_central_slice_width_with_error(self, padding: int = 10) -> tuple[float, float]:
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


def _f(measurement):
    return measurement.mean_central_slice_width_with_error(padding=10)


class TDSDispersionScan:
    def __init__(self, files: Iterable[os.PathLike]):
        self.measurements = [ScanMeasurement(df_path) for df_path in files]

    @property
    def dx(self) -> npt.NDArray:
        return np.array([s.dx for s in self.measurements])

    @property
    def tds(self) -> npt.NDArray:
        return np.array([s.tds for s in self.measurements])

    def max_energy_slice_widths_and_errors(self, padding: int = 20, do_mp: bool = False):
        if do_mp:
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


class DispersionScan(TDSDispersionScan):
    pass


class TDSScan(TDSDispersionScan):
    pass


def transform_pixel_widths(pixel_widths, pixel_widths_errors, pixel_units="px", to_variances=True):
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
    elif pixel_units == "um":
        scale = PIXEL_SCALE_X_UM
    elif pixel_units == "m":
        scale = PIXEL_SCALE_X_M
    else:
        raise ValueError(f"unknown unit width string: {pixel_units}")

    widths *= scale
    if to_variances:
        widths = widths**2

    # Extract errors form ufloats to go back to a 2-tuple of arrays.
    widths, errors = zip(*[(w.nominal_value, w.std_dev) for w in widths])
    return np.array(widths), np.array(errors)


def linear_fit(indep_var, dep_var, dep_var_err):
    popt, pcov = curve_fit(line, indep_var, dep_var, sigma=dep_var_err,
                           absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0]
    a1 = popt[1], perr[1]

    return a0, a1


def calculate_energy_spread_simple(scan: DispersionScan) -> tuple[float, float]:
    # Get measurement instance with highest dispresion for this scan
    dx, measurement = max((measurement.dx, measurement) for measurement in scan)
    energy = measurement.beam_energy  # in eV

    width_pixels = measurement.mean_central_slice_width_with_error()
    # Calculate with uncertainties automatically.
    width_pixels_unc = ufloat(*width_pixels)
    width_unc = width_pixels_unc * PIXEL_SCALE_X_M

    energy_spread_unc = energy * width_unc / dx
    energy_spread_ev = energy_spread_unc

    value, error = energy_spread_ev.n, energy_spread_ev.s

    return value, error  # in eV


@dataclass
class OpticalConfig:
    tds_length: float
    tds_voltages: list
    tds_wavenumber: float
    tds_bety: float
    tds_alfy: float
    ocr_betx: float

    @property
    def tds_gamy(self) -> float:
        return (1 + self.tds_alfy**2) / self.tds_bety


class SliceEnergySpreadMeasurement:
    def __init__(self, dscan: DispersionScan, tscan: TDSScan,
                 optical_config: OpticalConfig):
        self.dscan = dscan
        self.tscan = tscan
        self.oconfig = optical_config

        ntscan = len(self.tscan.measurements)
        nvoltages = len(self.oconfig.tds_voltages)

        if ntscan != nvoltages:
            raise RuntimeError("Mismatch between provided voltages and tscan measurements.")

    def dispersion_scan_fit(self) -> tuple[float, float]:
        widths, errors = self.dscan.max_energy_slice_widths_and_errors(padding=10)
        dx2 = self.dscan.dx**2
        # widths, errors = transform_units_for_pixel_widths(widths, errors)
        widths2_m2, errors2_m2 = transform_pixel_widths(widths, errors,
                                                        pixel_units="m", to_variances=True)
        a_v, b_v = linear_fit(dx2, widths2_m2, errors2_m2)
        return a_v, b_v

    def tds_scan_fit(self) -> tuple[float, float]:
        widths, errors = self.tscan.max_energy_slice_widths_and_errors(padding=10)
        voltages2 = self.oconfig.tds_voltages**2

        widths2_m2, errors2_m2 = transform_pixel_widths(widths, errors,
                                                        pixel_units="m", to_variances=True)
        a_v, b_v = linear_fit(voltages2, widths2_m2, errors2_m2)
        return a_v, b_v

    def all_fit_parameters(self) -> FittedBeamParameters:
        a_v, b_v = self.tds_scan_fit()
        a_d, b_d = self.dispersion_scan_fit()

        energy = self.tscan.beam_energy().mean() # in eV
        dispersion = self.tscan.dx.mean()

        return FittedBeamParameters(a_v=a_v, b_v=b_v, a_d=a_d, b_d=b_d,
                                    reference_energy=energy,
                                    reference_dispersion=dispersion,
                                    oconfig=self.oconfig)


@dataclass
class FittedBeamParameters:
    """Table 2 from the paper"""
    # Stored as tuples, nominal value with error.
    a_v: tuple[float, float]
    b_v: tuple[float, float]
    a_d: tuple[float, float]
    b_d: tuple[float, float]
    reference_energy: float
    reference_dispersion: float
    oconfig: OpticalConfig

    @property
    def sigma_e(self) -> float:
        energy0 = self.reference_energy
        dx0 = self.reference_dispersion
        # Convert to ufloat for correct error propagation before converting back
        # to tuples at the end.
        av = ufloat(*self.a_v)
        ad = ufloat(*self.a_d)
        result = ((energy0 / dx0) * umath.sqrt(av - ad))
        return result.n, result.s

    @property
    def sigma_i(self):
        """This is the average beamsize in the TDS, returned in metres"""
        k = self.oconfig.tds_wavenumber
        dx0 = self.reference_dispersion
        energy0 = self.reference_energy
        e0_joules = energy0 * e
        bv = ufloat(*self.b_v)
        result = (e0_joules / (dx0 * e * k)) * umath.sqrt(bv)
        return result.n, result.s

    @property
    def sigma_b(self):
        bety = self.oconfig.tds_bety
        alfy = self.oconfig.tds_alfy
        gamy = self.oconfig.tds_gamy
        length = self.oconfig.tds_length
        sigma_i = ufloat(*self.sigma_i)
        b_beta = (sigma_i**2
                  / (bety + 0.25 * length**2 * gamy - length * alfy))
        result = umath.sqrt(b_beta * self.oconfig.ocr_betx)
        return result.n, result.s

    @property
    def sigma_r(self):
        ad = ufloat(*self.a_d)
        sigma_b = ufloat(*self.sigma_b)
        result = umath.sqrt(ad - sigma_b**2)
        return result.n, result.s

    @property
    def emitx(self):
        gam0 = self.reference_energy / ELECTRON_MASS_EV
        sigma_b = ufloat(*self.sigma_b)
        result = sigma_b**2 * gam0 / self.oconfig.ocr_betx
        return result.n, result.s
