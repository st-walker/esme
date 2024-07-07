"""The image processing module"""

import cv2
import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndi
from scipy import LowLevelCallable
from uncertainties import ufloat
from scipy.optimize import curve_fit


import esme.maths as maths

from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

CENTRAL_SLICE_SEARCH_WINDOW_RELATIVE_WIDTH = 3

RawImageT = npt.NDArray


NOISE_THRESHOLD: float = 0.05

BACKGROUND_PIXEL_OUTLIER_FACTOR = 3


class ImageFunctor:
    def __init__(self, short="", long_description=""):
        self.short = short
        self.long_description = long_description

    def __call__(self, image):
        pass


class BackgroundSubtractor(ImageFunctor):
    def __init__(self, bg):
        self.bg = bg
        super().__init__(short="Background Subtracted",
                         long_description="Subtracts the background from the image and clips")

    def __call__(self, image):
        return subtract_background(image, self.bg)

class BackgroundNoiseRemover(ImageFunctor):
    def __init__(self, kernel_size=100):
        self.kernel_size = kernel_size
        super().__init__(short="Background Noise Smoothed and Masked",
                         long_description="Smooths the whole image and removes isolated, dim pixels")

    def __call__(self, image):
        return remove_background_noise(image, kernel_size=self.kernel_size)

class HotspotSmoother(ImageFunctor):
    def __init__(self):
        super().__init__(short="Hotspots Smoothed and Masked",
                         long_description="Smooths hotspot pixels and averages them with their neighbours")

    def __call__(self, image):
        return smooth_noise_hotspots(image)

class IsolatedPixelsRemover(ImageFunctor):
    def __init__(self):
        super().__init__(short="Isolated Pixel Clusters Removed",
                         long_description='Removes any pixels that are not attached to the main pixel "blob"')

    def __call__(self, image):
        return remove_all_disconnected_pixels(image)

class ImageCropper(ImageFunctor):
    def __init__(self):
        super().__init__(short="Image Cropped",
                         long_description='Crops the image by removing all consecutive rows and columns of zeroes starting from the bordersg ')

    def __call__(self, image):
        return remove_all_disconnected_pixels(image)


def make_image_processing_pipeline(bg):
    functors = [BackgroundSubtractor(bg),
                BackgroundNoiseRemover(kernel_size=100),
                HotspotSmoother(),
                IsolatedPixelsRemover()]
    return ImagePipeline(functors)
    


class ImagePipeline:
    def __init__(self, functors):
        self.functors = functors

    def __call__(self, image):
        for functor in self.functors:
            image = functor(image)
        # Copy input to at least keep metadata.
        return image

    def step_by_step(self, image, raw_image=False):
        if raw_image:
            yield raw_image
        for functor in self.functors:
            image = functor(image)
            yield image

    def __iter__(self):
        yield from iter(self.functors)




def mean8(image):
    @cfunc(intc(CPointer(float64), intp,
                CPointer(float64), voidptr))
    def _nbmean8(values_ptr, len_values, result, data):
        # Make array from inputs
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = 0
        for v in values:
            result[0] += v
        result[0] /= len_values
        return 1

    footprint = np.ones((3, 3))
    footprint[1, 1] = 0
    # generic_filter is unbelievably slow without using a LowLevelCallable
    # model="constant" and cval=0 because by definition beyond the
    # edges of the image in our case there is nothing (particularly if the image is cropped).
    # Use footprint of 3x3 1s with 0 in middle to not include central pixel.
    return ndi.generic_filter(image, LowLevelCallable(_nbmean8.ctypes),
                              footprint=footprint, mode='constant', cval=0.0)



class Image:
    def __init__(self, array, bg=0):
        self.array = array
        self.bg = bg

    def view(self):
        pass

def get_cropping_bounds(
        im: RawImageT, just_central_slices=False, sigma_tail_cut=3
) -> tuple[tuple[int, int], tuple[int, int]]:
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

    # if sigma_tail_cut:
    #     cut = 0.025 # 2 sigma cut (95% of )
    #     imc = im[irow0:irow1, icol0:icol1]
    #     cumsum = np.cumsum(imc.sum(axis=1)) / imc.sum()
    #     irow0 = irow0 + np.asarray(np.where(cumsum < cut)).max()
    #     irow1 = irow0 + np.asarray(np.where(cumsum > (1 - cut))).min()

    #     from IPython import embed; embed()

    # from IPython import embed; embed()

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

def get_beam_core_from_cropped_image(im0):
    # im0 should be cropped so that the whole beam is visible.
    pass


def get_cropping_slice(im: RawImageT) -> tuple:
    (row0, row1), (col0, col1) = get_cropping_bounds(im)
    return np.s_[row0: row1, col0: col1]


def crop_image(im: RawImageT) -> RawImageT:
    (idx_row0, idx_row1), (idx_col0, idx_col1) = get_cropping_bounds(im)
    return im[idx_row0:idx_row1, idx_col0:idx_col1]


def subtract_background(image, bg):
    """Subtract background and clip negative values"""
    return (image - bg).clip(min=0)


def remove_background_noise(image, kernel_size=100):
    # Apply uniform filter to try and smear out the big, isolated (background)
    # values
    im_filtered = ndi.uniform_filter(image, size=kernel_size)

    # Get mask for all pixels which, when smeared out, are below some max value
    # of the image. This should result in very isolated pixels getting set to 0,
    # and leave core beam pixels untouched, because they have many neighbours.
    mask = im_filtered < NOISE_THRESHOLD * im_filtered.max()
    image = image.copy()
    image[mask] = 0.0
    return image


def smooth_noise_hotspots(im0):
    # Get mean of each pixel and its surrounding 8 pixels, so 9 in total.
    img1 = ndi.uniform_filter(im0, size=3)

    # If a pixel is reduced by a lot in its brightness then that means
    # its neighbours are much dimmer than it, which presumably means
    # that this pixel is an outlier and should be removed.  The only
    # trick is to choose the constant factor.

    # In this case, we pick pixels which are 3x brighter than the average of it and its surrounding 8 pixels.
    # And we do it three times.
    for _ in range(5):
        mask = im0 > 3 * img1
        # And I should also do 3x darker?
        # mask = im0 > 3 * img1

        # For each individual masked point, we want to get the average
        # of its 8 surrounding pixels in the original, unmasked image.
        # This differs from uniform_filter only in that the centre
        # pixel is not included.
        means8 = mean8(im0)

        # Replace masked pixels with the mean of their surrounding 8 pixels
        im0[mask] = means8[mask]

    return im0


def filter_image(im0: RawImageT, bg: RawImageT, crop=False) -> RawImageT:
    im0bg = subtract_background(im0, bg)
    for _ in range(0, 3):
        im0bgm1 = remove_background_noise(im0bg)


    im0bgm2 = smooth_noise_hotspots(im0bgm1)
    im_no_outliers = remove_all_disconnected_pixels(im0bgm2)

    if crop:
        return crop_image(im_no_outliers)

    return im_no_outliers


def remove_all_disconnected_pixels(im: RawImageT) -> RawImageT:
    """Remove all pixels that are not attached to the main pixel
    "blob".  "Connected" means simply a sequence of adjacent non-zero
    pixels that attaches one pixel to the biggest blob of pixels
    (generally assumed to be the beam, otherwise something has gone wrong).

    """
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
    try:
        beam_blob_marker = np.argpartition(-counts, kth=1)[1]
    except:
        import ipdb; ipdb.set_trace()
    mask = markers == beam_blob_marker
    masked_image = np.where(mask, im, 0)

    return masked_image


def get_slice_properties(
        image: RawImageT, fast: bool = True, mask_nans=True,
        crop=True
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    #  Get bounds of image (i.e. to remove all fully-zero rows and columns)---this
    # speeds up the fitting procedure a lot by only fitting region of interest.
    irow0 = icol0 = 0
    irow1, icol1 = image.shape
    if crop:
        (irow0, irow1), (icol0, icol1) = get_cropping_bounds(
            image, just_central_slices=fast
        )
        # Do any cropping
        image = image[irow0:irow1, icol0:icol1]

    row_index = np.arange(image.shape[1])

    mean_slice_position = []
    mean_slice_position_error = []
    sigma_slice = []
    sigma_slice_error = []
    for (i,
        beam_slice
    ) in enumerate(image):  # Iterates over the ROWS, so each one is a slice of the beam.
        try:
            popt, perr = maths.get_gaussian_fit(row_index, beam_slice)
        except RuntimeError:  # Happens if curve_fit fails to converge.
            # Set parameters to NaN, mask them later from the output
            mu = sigma = sigma_mu = sigma_sigma = np.nan
        else:
            _, mu, sigma = popt
            _, sigma_mu, sigma_sigma = perr

        # if np.isnan(sigma):
                # mu = 0
                # sigma = 0
                # sigma_mu = 0
                # sigma_sigma = 0
        # except:
        #     import ipdb; ipdb.set_trace()

        mean_slice_position.append(mu)
        mean_slice_position_error.append(sigma_mu)
        sigma_slice.append(sigma)
        sigma_slice_error.append(sigma_sigma)

    # So we get back into the coordinate system of the original, uncropped image:
    row_index = np.arange(irow0, irow1)
    mean_slice_position += np.array(icol0)

    # Deal with nans due to failed fitting
    # I.e. where True = NaN
    nan_mask = (np.isnan(mean_slice_position)
                | np.isnan(mean_slice_position_error)
                | np.isnan(sigma_slice)
                | np.isnan(sigma_slice_error))


    # Deal with zeroes due to failed fitting or bad data, True == zero
    zeros_mask = ((mean_slice_position == 0)
                  | (mean_slice_position_error == 0)
                  | (sigma_slice == 0)
                  | (sigma_slice_error == 0))

    # # everything = row_index, mean_slice_position, slice_width
    # if np.isnan(mean_slice_position).any() or np.isnan(sigma_slice).any():
    #     from IPython import embed; embed()
    
    mean_slice_position = np.array(
        [ufloat(n, s) for n, s in zip(mean_slice_position, mean_slice_position_error)]
    )
    slice_width = np.array(
        [ufloat(n, s) for n, s in zip(sigma_slice, sigma_slice_error)]
    )

    if mask_nans:
        net_mask = nan_mask | zeros_mask
        row_index = row_index[~net_mask]
        mean_slice_position = mean_slice_position[~net_mask]
        slice_width = slice_width[~net_mask]

    return row_index, mean_slice_position, slice_width



def get_chosen_slice_from_fitted_parabola(rows, means):
    popt, _ = fit_parabola_to_image_means(rows, means)
    a, b, c = popt
    xmin = -b / (2 * a)
    return round(xmin)

def fit_parabola_to_image_means(rows, means):
    errors = [m.s for m in means]
    means = [m.n for m in means]
    return curve_fit(maths.parabola, rows, means, sigma=errors)


def get_fitted_parabola_from_image_means(rows, means):
    popt, _ = fit_parabola_to_image_means(rows, means)
    return maths.parabola(rows, *popt)

def get_gaussian_slice_properties(
        image: RawImageT, fast: bool = True, mask_nans=True,
        crop=True
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

    #  Get bounds of image (i.e. to remove all fully-zero rows and columns)---this
    # speeds up the fitting procedure a lot by only fitting region of interest.
    irow0 = icol0 = 0
    irow1, icol1 = image.shape
    if crop:
        (irow0, irow1), (icol0, icol1) = get_cropping_bounds(
            image, just_central_slices=fast
        )
        # Do any cropping
        image = image[irow0:irow1, icol0:icol1]

    row_index = np.arange(image.shape[1])

    mean_slice_position = []
    mean_slice_position_error = []
    sigma_slice = []
    sigma_slice_error = []
    for (i,
        beam_slice
    ) in enumerate(image):  # Iterates over the ROWS, so each one is a slice of the beam.
        try:
            popt, perr = maths.get_gaussian_fit(row_index, beam_slice)
        except RuntimeError:  # Happens if curve_fit fails to converge.
            # Set parameters to NaN, mask them later from the output
            mu = sigma = sigma_mu = sigma_sigma = np.nan
        else:
            _, mu, sigma = popt
            _, sigma_mu, sigma_sigma = perr

        try:
            if np.isnan(sigma):
                mu = 0
                sigma = 0
                sigma_mu = 0
                sigma_sigma = 0
        except:
            import ipdb; ipdb.set_trace()

        mean_slice_position.append(mu)
        mean_slice_position_error.append(sigma_mu)
        sigma_slice.append(sigma)
        sigma_slice_error.append(sigma_sigma)

    # So we get back into the coordinate system of the original, uncropped image:
    row_index = np.arange(irow0, irow1)
    mean_slice_position += np.array(icol0)

    # Deal with nans due to for example
    nan_mask = ~(
        np.isnan(mean_slice_position)
        | np.isnan(mean_slice_position_error)
        | np.isnan(sigma_slice)
        | np.isnan(sigma_slice_error)
    )

    mean_slice_position = np.array(
        [ufloat(n, s) for n, s in zip(mean_slice_position, mean_slice_position_error)]
    )
    slice_width = np.array(
        [ufloat(n, s) for n, s in zip(sigma_slice, sigma_slice_error)]
    )

    if mask_nans:
        row_index = row_index[nan_mask]
        mean_slice_position = mean_slice_position[nan_mask]
        slice_width = slice_width[nan_mask]

    return row_index, mean_slice_position, slice_width


def get_central_slice_width_from_slice_properties(means, sigmas, padding=10):
    centre_index = means.argmin()

    return np.mean(sigmas[centre_index - padding : centre_index + padding])

def get_average_central_slices(means, sigmas, padding=10):
    centre_index = np.argmin(means)
    return (np.mean(means[centre_index - padding : centre_index + padding]),
            np.mean(sigmas[centre_index - padding : centre_index + padding]))


def get_central_slice_width_from_parabola(means, sigmas, padding=10):
    pass


def get_selected_central_slice_width_from_slice_properties(means, sigmas, padding=10, slice_pos=None):
    centre_index = int(len(means)) // 2
    if slice_pos is None:
        centre_index = means.argmin()
    elif abs(slice_pos) > 0.5:
        raise ValueError("slice pos outside of [-0.5, 0.5].")
    else:
        centre_index += int(slice_pos * len(means) // 2)



    return centre_index, np.mean(sigmas[centre_index - padding : centre_index + padding])

def zero_off_axis_regions(image, xmin: int, xmax: int, ymin: int, ymax: int) -> None:
    image[:xmin] = 0.0
    image[xmax + 1:] = 0.0
    image[..., :ymin] = 0.0
    image[..., ymax + 1:] = 0.0

def clip_off_axis_regions(image, xmin: int, xmax: int, ymin: int, ymax: int) -> npt.NDArray:
    return image[ymin: ymax + 1, xmin: xmax + 1]
