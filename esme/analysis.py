from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import re
from pathlib import Path
from typing import Generator, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
from scipy.optimize import curve_fit
from uncertainties import ufloat

IMAGE_PATH_KEY = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"


NOISE_THRESHOLD = 0.08  # By eye...
PIXEL_SIZE_UM = 28  # 28 microns, from the paper

RawImageT = npt.NDArray

# def load_ims(screen_images: Iterable[TDSScreenImage]):
#     def f(tdsimage):
#         return tdsimage.to_im()

#     with mp.Pool(mp.cpu_count()) as p:
#         results = p.map(f, screen_images)

#     return results


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


def line(x, m, c):
    return m * x + c


def get_cropping_bounds(im: RawImageT) -> tuple[tuple[int, int], tuple[int, int]]:
    non_zero_column_indices = np.squeeze(np.where((im != 0).any(axis=0)))
    non_zero_row_indices = np.squeeze(np.where((im != 0).any(axis=1)))
    idx_column_start = non_zero_column_indices[0]
    idx_column_end = non_zero_column_indices[-1]
    idx_row_start = non_zero_row_indices[0]
    idx_row_end = non_zero_row_indices[-1]

    return (idx_row_start, idx_row_end), (idx_column_start, idx_column_end)


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
        return self.metadata["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"]


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

    def get_average_max_energy_slice_width(self, padding: int = 10) -> tuple[float, float]:
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
    return measurement.get_average_max_energy_slice_width(padding=10)


class TDSDispersionScan:
    def __init__(self, files: Iterable[os.PathLike]):
        self.measurements = [ScanMeasurement(df_path) for df_path in files]

    @property
    def dx(self) -> npt.NDArray:
        return np.array([s.dx for s in self.measurements])

    @property
    def tds(self) -> npt.NDArray:
        return np.array([s.tds for s in self.measurements])

    def get_max_energy_slice_widths(self, padding: int = 20, do_mp: bool = True):
        if do_mp:
            with mp.Pool(mp.cpu_count()) as p:
                return p.map(_f, self.measurements)
        return [_f(m) for m in self.measurements]

    def __getitem__(self, key: int) -> ScanMeasurement:
        return self.measurements[key]

    def beam_energy(self) -> float:
        return np.mean([m.beam_energy for m in self.beam_energy])

    def flatten(self, include_bg: bool = False) -> Generator[TDSScreenImage]:
        for measurement in self.measurements():
            yield from measurement.flatten(include_bg)


class DispersionScan(TDSDispersionScan):
    pass


class TDSScan(TDSDispersionScan):
    pass


def transform_variables_for_linear_fit(independent_variable, pixel_stds):
    """The fits used in the paper are linear relationships between the variances
    (i.e. pixel_std^2) and the square of the independent variable (either
    voltage squared V^2 or dipsersion D^2). This function takes D or V and sigma
    and transforms these variables to D^2 or V^2 and pixel width *variances* in
    units of um, so that the resulting fit for the standard deviations will be
    linear in the independent variable.

    """
    x2 = independent_variable**2
    # Do error calculation by converting to ufloats
    widths = np.array([ufloat(value, error) for (value, error) in pixel_stds])
    widths *= PIXEL_SIZE_UM  # Convert pixels to micrometres
    widths2 = widths**2
    # Extract errors
    widths2, errors2 = zip(*[(w.nominal_value, w.std_dev) for w in widths])
    return x2, widths2, errors2


def linear_fit_to_pixel_stds(indep_var, pixel_stds):
    # this function squarex both x and y, i.e. it assumes linear dependencey
    # beteen indep_var**2 and pixel_widths**2.
    x2, widths2, errors2 = transform_variables_for_linear_fit(indep_var, pixel_stds)

    # Fit with errors
    popt, pcov = curve_fit(line, x2, widths2, sigma=errors2, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Again convert to ufloats, and return
    c = ufloat(popt[1], perr[1])
    m = ufloat(popt[0], perr[0])

    return c, m


def plot_tds_scan(scan: TDSDispersionScan):
    widths = np.asarray(list(scan.get_max_energy_slice_widths(padding=10)))

    tds = scan.tds
    tds2 = tds**2

    # Do error calculation by converting to ufloats
    widths = np.array([ufloat(value, error) for (value, error) in widths])

    widths2 = widths**2
    errors2 = [x.s for x in widths2]
    widths2 = [x.n for x in widths2]

    popt, pcov = curve_fit(line, tds2, widths2, sigma=errors2, absolute_sigma=True)

    perr = np.sqrt(np.diag(pcov))

    m = ufloat(popt[0], perr[0])
    c = ufloat(popt[1], perr[1])

    tds2sample = np.linspace(0, 1.3 * max(tds2))
    sigma2fit = line(tds2sample, *popt)

    fig, ax = plt.subplots()
    ax.errorbar(tds2, widths2, yerr=errors2, label="Data")
    ax.plot(tds2sample, sigma2fit, label="Fit")
    ax.legend()

    ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{px}^2$")
    # ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{m}^2$")
    ax.set_xlabel(r"$TDS \%^2$")

    ax.set_title("TDS Scan Fit")

    return c, m



#     # dscan.show_before_after_for_measurement(-1)
#     # tdsscan.show_before_after_for_measurement(-1)    #
#     # # dscan[3].show(9)

#     plt.show()


# if __name__ == '__main__':
#     main()
