"""The image processing module"""

import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
import cv2
from uncertainties import ufloat

import esme.maths as maths


CENTRAL_SLICE_SEARCH_WINDOW_RELATIVE_WIDTH = 9

RawImageT = npt.NDArray


NOISE_THRESHOLD: float = 0.08  # By eye...


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
        length = irow1 - irow0
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


def get_slice_properties(image: RawImageT, fast: bool = True) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
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
    for beam_slice in imcropped:  # Iterates over the ROWS, so each one is a slice of the beam.
        try:
            popt, perr = maths.get_gaussian_fit(row_index, beam_slice)
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
    nan_mask = ~(
        np.isnan(mean_slice_position)
        | np.isnan(mean_slice_position_error)
        | np.isnan(sigma_slice)
        | np.isnan(sigma_slice_error)
    )

    mean_slice_position = np.array([ufloat(n, s) for n, s in zip(mean_slice_position, mean_slice_position_error)])
    slice_width = np.array([ufloat(n, s) for n, s in zip(sigma_slice, sigma_slice_error)])

    row_index = row_index[nan_mask]
    mean_slice_position = mean_slice_position[nan_mask]
    slice_width = slice_width[nan_mask]

    return row_index, mean_slice_position, slice_width

def get_central_slice_width_from_slice_properties(means, sigmas, padding=10):
    centre_index = means.argmin()
    return np.mean(sigmas[centre_index - padding : centre_index + padding])
