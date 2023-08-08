import numpy as np
from scipy.constants import e

from esme.analysis import (
    ParameterScan,
    ScanMeasurement,
    SliceEnergySpreadMeasurement,
    transform_pixel_widths,
)
from esme.calibration import TDS_WAVENUMBER, r34s_from_scan
from esme.image import crop_image
from esme.maths import get_gaussian_fit


def bunch_lengths_from_scan_measurement(measurement: ScanMeasurement, pixel_units="m"):
    lengths = []
    errors = []
    for i in range(measurement.nimages):
        image = measurement.to_im(i)
        image = crop_image(image)
        pixel_indices = np.arange(image.shape[0])  # Assumes streaking is in image Y
        projection = image.sum(axis=1)
        popt, perr = get_gaussian_fit(pixel_indices, projection)
        sigma = popt[2]
        # sigma = np.sqrt(np.cov(pixel_indices, aweights=projection))

        lengths.append(sigma)

    mean_length = np.mean(lengths)
    mean_error = mean_length / np.sqrt(len(lengths))

    # Transform units from px to whatever was chosen
    mean_length, mean_error = transform_pixel_widths(
        [mean_length],
        [mean_error],
        to_variances=False,
        pixel_units=pixel_units,
        dimension="y",
    )
    return mean_length, mean_error


def apparent_bunch_lengths(scan: ParameterScan):
    average_lengths = []
    average_lengths_errors = []
    for measurement in scan:
        mean_size, mean_error = bunch_lengths_from_scan_measurement(measurement)
        average_lengths.append(mean_size.item())
        average_lengths_errors.append(mean_error.item())
    return np.array(average_lengths), np.array(average_lengths_errors)


def true_bunch_lengths(scan: ParameterScan, voltages=None):
    raw_bl, raw_bl_err = apparent_bunch_lengths(scan)
    r34s = abs(r34s_from_scan(scan))
    if voltages is None:
        voltages = abs(scan.voltage)
    energy = scan.beam_energy() * e

    true_bl = (energy / (e * voltages * TDS_WAVENUMBER)) * raw_bl / r34s
    true_bl_err = (energy / (e * voltages * TDS_WAVENUMBER)) * raw_bl_err / r34s

    return true_bl, true_bl_err


def mean_bunch_length_from_time_calibration(esme: SliceEnergySpreadMeasurement):
    lengths = []
    uncertainties = []

    if esme.dscan is not None:
        dlengths, dlengths_unc = true_bunch_lengths(esme.dscan)
        lengths.extend(dlengths)
        uncertainties.extend(dlengths_unc)

    if esme.tscan is not None:
        tlengths, tlengths_unc = true_bunch_lengths(esme.tscan)
        lengths.extend(tlengths)
        uncertainties.extend(tlengths_unc)

    if esme.bscan is not None:
        blengths, blengths_unc = true_bunch_lengths(esme.bscan)
        lengths.extend(blengths)
        uncertainties.extend(blengths_unc)

    return _mean_with_uncertainties(lengths, uncertainties)


def _mean_with_uncertainties(values, stdevs):
    # Calculate mean of n values each with 1 uncertainty.
    assert len(values) == len(stdevs)
    mean = np.mean(values)
    variances = np.power(stdevs, 2)
    mean_stdev = np.sqrt(np.sum(variances) / (len(values) ** 2))
    return mean, mean_stdev
