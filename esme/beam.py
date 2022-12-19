import numpy as np
from scipy.constants import e

from esme.analysis import (ScanMeasurement, ParameterScan, crop_image,
                           transform_pixel_widths, get_gaussian_fit, gauss)
from esme.calibration import r34s_from_scan, TDS_WAVENUMBER


def bunch_lengths_from_scan_measurement(measurement: ScanMeasurement, pixel_units="m"):
    lengths = []
    errors = []
    for i in range(measurement.nimages):
        image = measurement.to_im(i)
        image = crop_image(image)
        pixel_indices = np.arange(image.shape[0]) # Assumes streaking is in image Y
        projection = image.sum(axis=1)
        popt, perr = get_gaussian_fit(pixel_indices, projection)
        sigma = popt[2]
        # sigma = np.sqrt(np.cov(pixel_indices, aweights=projection))


        lengths.append(sigma)

    mean_length = np.mean(lengths)
    mean_error = mean_length / np.sqrt(len(lengths))
    
    # Transform units from px to whatever was chosen
    mean_length, mean_error = transform_pixel_widths([mean_length], [mean_error],
                                                     to_variances=False,
                                                     pixel_units=pixel_units,
                                                     dimension="y")
    # from IPython import embed; embed()
    
    return mean_length, mean_error


def apparent_bunch_lengths(scan: ParameterScan):
    average_lengths = []
    average_lengths_errors = []
    for measurement in scan:
        mean_size, mean_error = bunch_lengths_from_scan_measurement(measurement)
        average_lengths.append(mean_size.item())
        average_lengths_errors.append(mean_error.item())
    return np.array(average_lengths), np.array(mean_error)


def true_bunch_lengths(scan: ParameterScan):
    raw_bl, raw_bl_err = apparent_bunch_lengths(scan)
    r34s = abs(r34s_from_scan(scan))
    voltages = abs(scan.voltage)
    energy = scan.beam_energy() * e
    
    true_bl = (energy / (e * voltages * TDS_WAVENUMBER)) * raw_bl / r34s
    true_bl_err = (energy / (e * voltages * TDS_WAVENUMBER)) * raw_bl_err / r34s

    return true_bl, true_bl_err

# , raw_bl_err / r34s
