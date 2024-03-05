"""Set of functions for plotting and make results tables"""


from dataclasses import dataclass
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import tabulate
from uncertainties import ufloat
from scipy.ndimage import gaussian_filter1d

import esme.calibration as cal
import esme.image as image
# import esme.lattice as lat
import esme.maths as maths
import esme.analysis as ana
import esme.optics as optics


LOG = logging.getLogger(__name__)

SLICE_WIDTH_LABEL = r"$\sigma_M\,/\,\mathrm{\mu m}$"
SLICE_WIDTH_PIXELS_LABEL = r"$\sigma_M\,/\,\mathrm{px}$"
BETA_LABEL = r"$\beta_\mathrm{OTR}\,/\,\mathrm{m}$"
ETA_LABEL = r"$\eta_\mathrm{{OTR}}\,/\,\mathrm{m}$"
ETA2_LABEL = r"$\eta^2_\mathrm{{OTR}}\,/\,\mathrm{m}$"
VOLTAGE_LABEL = r"$|V_\mathrm{TDS}|\,/\,\mathrm{MV}$"
VOLTAGE2_LABEL = r"$|V^2_\mathrm{TDS}|\,/\,\mathrm{MV}$"
TDS_CALIBRATION_LABEL = r"Gradient / $\mathrm{\mu{}mps^{-1}}$"
TDS_AMPLITUDE_SETPOINT_LABEL = r"TDS Amplitude Setpoint / %"
TDS_AMPLITUDE_READBACK_LABEL = r"TDS Amplitude Readback / %"
TDS_AMPLITUDE_LABEL = "TDS Amplitude / %"
STREAKING_LABEL = "$S$"
R34_LABEL = r"$R_{34}\,/\,\mathrm{mm\cdot{}mrad^{-1}}$"
BEAM_SIZE_Y_LABEL = r"$\sigma_y\,/\,\mathrm{\mu m}$"
LONG_RESOLUTION_LABEL = r"$(\sigma_y\,/\,S)\,/\,\mathrm{\mu{}m}$"
LONG_RESOLUTION_PIXELS_LABEL = r"$(\sigma_y\,/\,S)\,/\,\mathrm{px}$"

def break_single_image_down(im, strategy, title="", fast_analysis=True):
    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 2, 1) # Image axes
    ax2 = fig.add_subplot(2, 2, 2, sharey=ax1) #
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    index_chosen_row = strategy.slice_centre

    # First show the image and the chosen row that we are using for
    # the maximum slice, this is pre-calculated using the fully
    # processed image, and then used for the (maybe raw) image here
    ax1.imshow(im)
    ax1.axhline(index_chosen_row, alpha=0.25, color="white")

    # The image should already be cropped, so we don't crop it again.
    # Do the slice analysis on the image now which should be cropped
    # but otherwise is in some generally unknown position along the
    # image processing pipeline.
    rows, means, sigmas = image.get_slice_properties(im, fast=fast_analysis, mask_nans=False, crop=False)
    parab_fit_mask = means.nonzero() # Filter where rows or means are zero
    rows_parab = rows[parab_fit_mask]
    means_parab = means[parab_fit_mask]
    yparab = image.get_fitted_parabola_from_image_means(rows_parab, means_parab)

    # Annoying crap to get rid of the uncertainties
    means = [m.n for m in means]
    sigmas = [s.n for s in sigmas]
    # Get the indices for inverted plotting
    indices = list(range(len(means)))
    # Plot the mean slice position on the beam image.
    ax1.plot(means, indices, linewidth=0.4, color="green")

    ax4.plot(means, indices, color="green", label="Gaussian means")


    # Plot to the right of the image the slice widths for the (maybe
    # raw!) image
    ax2.plot(sigmas, indices, label="Slice Widths from image")

    # Now we filter them to provide a trend line on top of the raw widths.
    # ax2.plot(sigmasf, indices, label="Smoothed Slice Widths")
    # centre_index_when_smoothed = np.argmin(meansf)

    window = strategy.window
    lower, upper = get_bounds_from_window(index_chosen_row, window)

    # Show region that I am using
    ax2.axhspan(lower, upper, color="orange", alpha=0.25, label=f"{strategy.window} Slices about $E_\mathrm{{max}}$")

    # ax2.axhline(rows[centre_index_when_smoothed], alpha=0.25, color="pink", label="Peak Energy Slice if smoothed")
    ax1.plot(yparab, rows_parab, label="Parabolic fit", color="red", linewidth=0.4)

    ax4.plot(yparab, rows_parab, color="red", label="Parabolic fit to Gaussian means")

    ax4.legend()
    ax4.set_ylabel("Row Index")
    ax4.set_xlabel("Column Index")

    ax4.set_title("Slice means with parabolic fits")

    ax2.set_xlim(3, 6)

    # Plot the central  slice's projection
    ax3.plot(im[index_chosen_row], label="Peak Energy Row")


    # Boring bits
    # Ax1
    ax1.set_title("Cropped Image")
    # Ax2
    ax2.set_title("Slice sigmas with/without Gaussian filter.")
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.set_xlabel(r"$\sigma_M$ / px")
    ax2.set_ylabel("Row Index")
    ax2.legend()
    # Ax3
    ax3.legend()
    ax3.set_title(fr"$\sigma_M = {sigmas[index_chosen_row]}$ px")
    ax3.yaxis.tick_right()

    # Fig
    fig.suptitle(title)

    return ImageAnalysisResult(figure=fig,
                               central_slice_index=index_chosen_row,
                               window=window,
                               widths=sigmas)


def get_bounds_from_window(centre, window_full_size):
    slice_half_width = window_full_size // 2
    lower = centre - slice_half_width
    upper = centre + slice_half_width + 1
    return lower, upper


def plot_all_slice_widths(measurement):
    setpoints = measurement.tscans
    fig, ax = plt.subplots()
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = prop_cycle.by_key()['color']
    # setpoints = setpoints[1:2]

    for (setpoint, colour) in zip(setpoints, colours):
        label = f"V = {setpoint.voltage / 1e6} MV"
        for path in setpoint.image_full_paths:
            im = np.load(path)["image"]
            im = im.T
            # Do "canonical"/standard image analysis to refer back to to get the widths
            # XXX: NO BACKGROUND ?
            processed_image = image.filter_image(im, bg=0.0)
            crop = image.get_cropping_slice(processed_image)
            # These will be in global coordinates
            rows_processed, means_processed, bunch_sigmas_processed = image.get_slice_properties(processed_image[crop], fast=False)

            # Means and sigmas here are for the unrpocessed image, so
            # it is unwise to use the means here to pick the max energy slice.
            rows, means, bunch_sigmas = image.get_slice_properties(im[crop], fast=fast_analysis)
            # I need to instead

            bunch_sigmas = [b.n for b in bunch_sigmas]
            # means = gaussian_filter1d(means, 4)
            bunch_sigmas = gaussian_filter1d(bunch_sigmas, 4)

            # from IPython import embed; embed()
            centre_index = means_processed.argmin()
            rows = rows - rows[centre_index]

            ax.plot(rows, bunch_sigmas, label=label, color=colour, linewidth=0.2)
            label = None

    ax.set_xlabel("Slice pixel index from peak energy slice")
    ax.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax.legend()
    plt.show()


def break_setpoint_image_analysis_down(setpoint, outdir, fast_analysis=True, window=21, parabolic=False) -> None:
    outdir = Path(str(Path(outdir) / setpoint.name).replace(" ",""))
    outdir.mkdir(exist_ok=True, parents=True)

    fast_analysis = True

    all_widths = []
    all_window_stds = []
    # for i, (im, bg) in enumerate(setpoint.get_images_with_background()):
    pipeline = make_image_processing_pipeline(bg=setpoint.bg)

    for i, tim in enumerate(setpoint.tagged_images()):
        dbim = DecomposedBeamImage(pipeline, tim)

        sliceana = dbim.slice_analysis

        if parabolic:
            chosen_row_index = image.get_chosen_slice_from_fitted_parabola(sliceana.rows,
                                                                           sliceana.means)
        else:
            centre_index = sliceana.means.argmin()
            chosen_row_index = aliceana.rows[centre_index] # in original uncropped coordinate system


        filebasename = f"image-{i}"

        method = "mean"
        if parabolic:
            method = "parabolic"

        strategy = SlicingStrategy(slice_centre=chosen_row_index,
                                   window=window,
                                   method=method)

        # I process the cropped image.
        widths, window_width_stds = break_filter_image_down(dbim,
                                                            strategy,
                                                            outdir,
                                                            filebasename,
                                                            fast_analysis=fast_analysis)
        all_widths.append(widths)
        all_window_stds.append(window_width_stds)

    # Finally we do a bit of filtering on the final step of the
    # widths.  We do this here and not for each individual image
    # because we need the full sample to determine if there are outliers or not.
    all_widths = np.array(all_widths)
    all_window_stds = np.array(all_window_stds)
    outliers_mask = ana.make_outlier_widths_mask(all_widths[..., -1], 3)[..., np.newaxis]

    filtered_widths = all_widths[..., -1, np.newaxis].copy()
    filtered_window_stds = all_window_stds[..., -1, np.newaxis].copy()

    filtered_widths[outliers_mask] = np.nan
    filtered_window_stds[outliers_mask] = np.nan

    all_widths = np.hstack((all_widths, filtered_widths))
    all_window_stds = np.hstack((all_window_stds, filtered_window_stds))

    fig, ax = plt.subplots()
    for window_std in all_window_stds:
        ax.plot(STAGE_MAP, window_std)
    mean_window_std_by_image_processing_step = np.nanmean(all_window_stds, axis=0)
    ax.set_xlabel("Image Processing Stage")
    ax.set_ylabel(r"$\sigma_{\sigma_M}$ / px")
    ax.set_title(f"Window size = {window}")
    ax.plot(STAGE_MAP, mean_window_std_by_image_processing_step, label="Average", linewidth=4.0, linestyle="--", color="black")
    ax.set_ylim(np.nanmin(all_window_stds[..., -1]) * 0.9, np.nanmax(all_window_stds[..., -1]) / 0.9)
    fig.savefig(outdir / "./all-image-processing-step-noise.pdf")


    fig, ax = plt.subplots()
    for widths in all_widths:
        ax.plot(STAGE_MAP, widths)

    bg = 0.0
    ax.set_xlabel("Image Processing Stage")
    ax.set_ylabel(SLICE_WIDTH_PIXELS_LABEL)

    if bg == 0.0:
        ax.set_title("No Background Subtraction done")
    else:
        ax.set_title("With background subtraction")

    mean_widths_by_image_processing_step = np.nanmean(all_widths, axis=0)
    ax.plot(STAGE_MAP, mean_widths_by_image_processing_step, label="Average", linewidth=4.0, linestyle="--", color="black")
    ax.legend()

    ax.set_title(f"Average $\sigma_M$ from {min(mean_widths_by_image_processing_step)} px to {max(mean_widths_by_image_processing_step)} px")
    fig.savefig(outdir / "./all-image-processing-step-widths.pdf")

    widths_fully_processed = all_widths[..., -1]
    widths_fully_processed = widths_fully_processed[~np.isnan(widths_fully_processed)]
    plt.close("all")
    final_mean_width = np.nanmean(widths_fully_processed)
    final_mean_window_noise = mean_window_std_by_image_processing_step[-1]

    return final_mean_width, final_mean_window_noise


@dataclass
class ImageAnalysisResult:
    figure: plt.Figure
    central_slice_index: int
    window: int
    widths: np.ndarray

    @property
    def window_slice(self):
        lower, upper = get_bounds_from_window(self.central_slice_index, self.window)
        return np.s_[lower: upper]

    @property
    def width(self):
        return np.mean(self.window_widths)

    @property
    def window_widths(self):
        return self.widths[self.window_slice]


# @dataclass
# class SetpointImageAnalysisResult

STAGE_MAP = ["Raw Image",
             "Background Subtraction",
             "Uniform Filtered: 100",
             "Uniform Filtered: 3",
             "Remove isolated pixels",
             "Outlier images filtered"]


@dataclass
class SlicingStrategy:
    slice_centre: int
    window: int = 21
    method: str = "mean"


def plot_decomposed_image(im_decomp, strategy, outdir, filebasename, fast_analysis=True, window=21):
    outdir = Path(outdir)

    stage = 0
    widths = []
    stages = []
    window_stds = []

    
    for im in im_decomp.all_stages():
        break_single_image_down(im, strategy, title="Raw Image", fast_analysis=fast_analysis)
        
    
    res = break_single_image_down(im, strategy, title="Raw Image", fast_analysis=fast_analysis)
    save_and_close(res.figure, outdir / f"{filebasename}-stage-{stage}.pdf")
    stages.append(stage)
    widths.append(res.width)
    window_stds.append(np.std(res.window_widths))
    stage +=1

    im = image.subtract_background(im, bg)
    res = break_single_image_down(im, strategy, title="Background subtracted image", fast_analysis=fast_analysis)
    save_and_close(res.figure, outdir / f"{filebasename}-stage-{stage}.pdf")
    stages.append(stage)
    widths.append(res.width)
    window_stds.append(np.std(res.window_widths))
    stage += 1

    size = 100
    im = image.remove_background_noise(im, size=size)
    res = break_single_image_down(im, strategy, title=f"Background noise removed", fast_analysis=fast_analysis)
    save_and_close(res.figure, outdir / f"{filebasename}-stage-{stage}.pdf")
    stages.append(stage)
    widths.append(res.width)
    window_stds.append(np.std(res.window_widths))
    stage += 1

    size = 3
    im = image.smooth_noise_hotspots(im)
    res = break_single_image_down(im, strategy, title=f"Hotspot pixels smoothed", fast_analysis=fast_analysis)
    save_and_close(res.figure, outdir / f"{filebasename}-stage-{stage}.pdf")
    stages.append(stage)
    widths.append(res.width)
    window_stds.append(np.std(res.window_widths))
    stage += 1

    im = image.remove_all_disconnected_pixels(im)
    res = break_single_image_down(im, strategy, title=f"With Isolated blobs removed.  (Final image step)", fast_analysis=fast_analysis)
    save_and_close(res.figure, outdir / f"{filebasename}-stage-{stage}.pdf")
    stages.append(stage)
    widths.append(res.width)
    window_stds.append(np.std(res.window_widths))
    stage += 1

    return widths, window_stds


def save_and_close(figure, path):
    figure.savefig(path)
    figure.clf()
    plt.close(figure)
    plt.close("all")
    plt.close()


def plot_dispersion_scan(esme, ax=None) -> None:
    scan = esme.dscan
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    dx2 = scan.dx**2

    widths_um2, errors_um2 = ana.transform_pixel_widths(
        widths, errors, pixel_units="um", to_variances=True
    )
    a0, a1 = maths.linear_fit(dx2, widths_um2, errors_um2)

    d2sample = np.linspace(0, 1.1 * max(dx2))
    sigma2fit = maths.line(d2sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(
        dx2, widths_um2, yerr=errors_um2, label="Data", marker=".", linestyle=""
    )
    ax.plot(d2sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$D^2\,/\,\mathrm{m}^2$")
    ax.set_title("Dispersion scan fit")
    add_info_box(ax, "D", "m", a0, a1)


def _set_ylabel_for_scan(ax):
    ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{\mathrm{\mu}m}^2$")
    # ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{m}^2$")
    # ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{px}^2$")


def write_pixel_widths(esme, outdir):
    dwidths, derrors = esme.dscan.max_energy_slice_widths_and_errors(padding=10)
    twidths, terrors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    dwidths_um, derrors_um = ana.transform_pixel_widths(
        dwidths, derrors, pixel_units="um", to_variances=False
    )
    twidths_um, terrors_um = ana.transform_pixel_widths(
        twidths, terrors, pixel_units="um", to_variances=False
    )
    dscan_data = pd.DataFrame(
        {
            "dx": esme.dscan.dx,
            "amplitude": esme.dscan.tds_percentage,
            "voltage": esme.dscan.voltage,
            "px_widths": dwidths,
            "px_errors": derrors,
            "um_widths": dwidths_um,
            "um_errors": derrors_um,
        }
    )

    tds_data = pd.DataFrame(
        {
            "dx": esme.tscan.dx,
            "amplitude": esme.tscan.tds_percentage,
            "voltage": esme.tscan.voltage,
            "px_widths": twidths,
            "px_errors": terrors,
            "um_widths": twidths_um,
            "um_errors": terrors_um,
        }
    )

    dscan_data.to_csv(outdir / "dscan_central_slices.csv")
    tds_data.to_csv(outdir / "tscan_central_slices.csv")


def plot_tds_scan(esme, ax=None) -> None:
    widths, errors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    voltages_mv = _get_tds_tscan_abs_voltage_in_mv_from_scans(esme)

    voltages2_mv2 = voltages_mv**2
    widths_um2, errors_um2 = ana.transform_pixel_widths(
        widths, errors, pixel_units="um"
    )

    a0, a1 = maths.linear_fit(voltages2_mv2, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    v2_sample = np.linspace(0.9 * min(voltages2_mv2), 1.1 * max(voltages2_mv2))
    sigma2fit = maths.line(v2_sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(
        voltages2_mv2,
        widths_um2,
        yerr=errors_um2,
        label="Data",
        marker=".",
        linestyle="",
    )
    ax.plot(v2_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Voltage}^2\,/\,\mathrm{MV}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "MV", a0, a1)


def plot_beta_scan(esme, ax=None) -> None:
    widths, errors = esme.bscan.max_energy_slice_widths_and_errors(padding=10)

    beta = esme.bscan.beta

    widths_um2, errors_um2 = ana.transform_pixel_widths(
        widths, errors, pixel_units="um"
    )

    a0, a1 = maths.linear_fit(beta, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    beta_sample = np.linspace(0.9 * min(beta), 1.1 * max(beta))
    sigma2fit = maths.line(beta_sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.errorbar(
        beta, widths_um2, yerr=errors_um2, label="Data", marker=".", linestyle=""
    )
    ax.plot(beta_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\beta_\mathrm{OCR}\,/\,\mathrm{m}$")
    ax.set_title("Beta scan fit")
    add_info_box(ax, r"\beta", "m", a0, a1)


def plot_scans(esme, root_outdir=None) -> None:
    if esme.bscan:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14, 8))
        plot_beta_scan(esme, ax3)
        figb, axb = plt.subplots()
        plot_beta_scan(esme, axb)
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 8))

    figd, axd = plt.subplots()
    figt, axt = plt.subplots()
    plot_dispersion_scan(esme, ax1)
    plot_dispersion_scan(esme, axd)
    plot_tds_scan(esme, ax2)
    plot_dispersion_scan(esme, axt)

    fig.suptitle("Dispersion and TDS Scans for an Energy Spread Measurement")

    if root_outdir is not None:
        fig.savefig(root_outdir / "scan-fits.png")
    else:
        plt.show()


def add_info_box(ax, symbol, xunits, c, m) -> None:
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    textstr = '\n'.join(
        [
            rf"$\sigma_M^2 = A_{{{symbol}}} +B_{{{symbol}}} {{{symbol}}}^2$",
            rf"$A_{{{symbol}}} = ({c[0]:.2f}\pm{c[1]:.1g})\,\mathrm{{\mu m}}^2$",
            rf"$B_{{{symbol}}} = ({m[0]:.2f}\pm{m[1]:.1g})\,\mathrm{{\mu m^2\,/\,{xunits}}}^2$",
        ]
    )

    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment='top',
        bbox=props,
    )


def plot_measured_central_widths(
    esme,
    root_outdir=None,
    show=True,
    write_widths=True,
) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    dscan = esme.dscan
    tscan = esme.tscan
    voltages = esme.tscan.tds_percentage

    dx = dscan.dx

    if dscan.measurements:
        dwidths, derrors = dscan.max_energy_slice_widths_and_errors(padding=10)
        dwidths_um, derrors_um = ana.transform_pixel_widths(
            dwidths, derrors, pixel_units="um", to_variances=False
        )
        plot_scan_central_widths(dscan, dx, ax1, ax3)

        if write_widths:
            data = {
                "dx": dx,
                "px_widths": dwidths,
                "px_errors": derrors,
                "um_widths": dwidths_um,
                "um_errors": derrors_um,
            }
            pd.DataFrame.from_dict(data).to_pickle("dscan-pixels.pcl")

    if tscan.measurements:
        twidths, terrors = tscan.max_energy_slice_widths_and_errors(padding=10)
        twidths_um, terrors_um = ana.transform_pixel_widths(
            twidths, terrors, pixel_units="um", to_variances=False
        )
        plot_scan_central_widths(tscan, voltages, ax2, ax4)

        if write_widths:
            data = {
                "voltage": voltages,
                "px_widths": twidths,
                "px_errors": terrors,
                "um_widths": twidths_um,
                "um_errors": terrors_um,
            }
            pd.DataFrame.from_dict(data).to_pickle("tscan-pixels.pcl")

    ax1.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax3.set_ylabel(r"$\sigma_M\,/\,\mathrm{\mu m}$")
    ax3.set_xlabel("D / m")
    # ax4.set_xlabel("TDS Voltage / MV")
    ax4.set_xlabel("TDS Amplitude / %")

    fig.suptitle(
        fr"Measured maximum-energy slice widths for pixel scale X = {ana.PIXEL_SCALE_X_UM} $\mathrm{{\mu m}}$"
    )

    if show:
        plt.show()

    if root_outdir is not None:
        fig.savefig(root_outdir / "measured-central-widths.png")


def plot_scan_central_widths(scan, x, ax1, ax2):
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    widths_um, errors_um = ana.transform_pixel_widths(
        widths, errors, pixel_units="um", to_variances=False
    )

    ax1.errorbar(x, widths, yerr=errors, marker="x")
    ax2.errorbar(x, widths_um, yerr=errors_um, marker="x")


# def _um_tuple(pair):
#     return pair*-

def break_scan_down(setpoints, outdir, xvar, xlabel, scan_type, window=21, parabolic=False):
    widths = []
    window_stds = []
    outdir /= scan_type
    outdir.mkdir(exist_ok=True, parents=True)

    for setpoint in setpoints:
        mean_width, mean_window_std = break_setpoint_image_analysis_down(setpoint, outdir, window=window, parabolic=parabolic)
        widths.append(mean_width)
        window_stds.append(mean_window_std)

    # from IPython import embed; embed()

    widths = np.array(widths)
    window_stds = np.array(window_stds)
    # from IPython import embed; embed()

    xvar_widths = np.hstack((xvar[..., np.newaxis], widths[..., np.newaxis]))
    xvar_stds = np.hstack((xvar[..., np.newaxis], window_stds[..., np.newaxis]))

    np.savetxt(outdir / f"{scan_type}-widths.txt", xvar_widths)
    np.savetxt(outdir/ f"{scan_type}-window-stds.txt", xvar_stds)

    fig, ax = plt.subplots()
    ax.plot(xvar, widths, marker="x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sigma_M$ / px")
    fig.savefig(outdir / f"{scan_type}-mean-slice-widths.pdf")

    fig, ax = plt.subplots()
    ax.plot(xvar, window_stds, marker="x")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\sigma_{\sigma_M}$ / px")
    fig.savefig(outdir / f"{scan_type}-mean-window-stds.pdf")


def compare_both_derivations(test, latex=True):
    pass


# def _derived_paramaters(params):


def _coefficients_df(params):
    pass


def formatted_parameter_dfs(params, latex=False) -> str:
    # params = esme.all_fit_parameters()

    fit_params = params.fit_parameters_to_df()

    fit_params.loc[["V_0", "E_0"]] *= 1e-6  # To MV / MeV

    beam_params = params.beam_parameters_to_df()

    # from IPython import embed; embed()

    if params.sigma_z is not None:
        beam_params.loc["sigma_z"] = {"values": params.sigma_z[0], "errors": params.sigma_z[1]}
        beam_params.loc["sigma_z"] *= 1e3  # to mm
        # To picseconds:
        beam_params.loc["sigma_t"] = {"values": params.sigma_t[0]*1e12,
                                      "errors": params.sigma_t[1]*1e12}

    if params.sigma_z_rms is not None:
        beam_params.loc["sigma_z_rms"] = {"values": params.sigma_z_rms[0], "errors": params.sigma_z_rms[1]}
        beam_params.loc["sigma_z_rms"] *= 1e3  # to mm
        # To picseconds:
        beam_params.loc["sigma_t_rms"] = {"values": params.sigma_t_rms[0]*1e12,
                                          "errors": params.sigma_t_rms[1]*1e12}

        
    beam_params.loc[
        ["emitx", "sigma_i", "sigma_b", "sigma_r"]
    ] *= 1e6  # to mm.mrad & um
    beam_params.loc[["sigma_e", "sigma_e_from_tds"]] *= 1e-3  # to keV

    units = {
        'V_0': 'MV',
        'D_0': 'm',
        'E_0': 'MeV',
        'A_V': 'm2',
        'B_V': 'm2/V2',
        'A_D': 'm2',
        'B_D': '',
        'A_beta': 'm2',
        'B_beta': 'm',
        "sigma_z": "mm",
        "sigma_t": "ps",
        "sigma_z_rms": "mm",
        "sigma_t_rms": "ps",
        'sigma_e': 'keV',
        'sigma_e_from_tds': 'keV',
        'sigma_i': 'um',
        'sigma_b': 'um',
        'sigma_r': 'um',
        'emitx': 'mm.mrad',
    }

    latex_variables = {
        'V_0': '$V_0$',
        'D_0': '$D_0$',
        'E_0': '$E_0$',
        'A_V': '$A_V$',
        'B_V': '$B_V$',
        'A_D': '$A_D$',
        'B_D': '$B_D$',
        'A_beta': '$A_\\beta$',
        'B_beta': '$B_\\beta$',
        "sigma_z": r"$\sigma_z$",
        "sigma_t": r"$\sigma_t$",
        "sigma_z_rms": r"$\sigma_z^\mathrm{rms}$",
        "sigma_t_rms": r"$\sigma_t^\mathrm{rms}$",
        'sigma_e': '$\\sigma_E$',
        'sigma_i': '$\\sigma_I$',
        'sigma_e_from_tds': '$\\sigma_E^{\mathrm{TDS}}$',
        'sigma_b': '$\\sigma_B$',
        'sigma_r': '$\\sigma_R$',
        'emitx': '$\\varepsilon_x$',
    }

    latex_units = {
        "m2": r"\si{\metre\squared}",
        "MV": r"\si{\mega\volt}",
        "m": r"\si{\metre}",
        "mm": r"\si{\milli\metre}",
        "um": r"\si{\micro\metre}",
        "m2/V2": r"\si{\metre\squared\per\volt\squared}",
        "keV": r"\si{\kilo\electronvolt}",
        "MeV": r"\si{\mega\electronvolt}",
        "ps": r"\si{\pico\second}",
        "": "",
        "mm.mrad": r"\si{\milli\metre{}\cdot{}\milli\radian}",
    }

    varnames = None

    if latex:
        varnames = latex_variables
        units = {
            var_name: latex_units[unit_str] for (var_name, unit_str) in units.items()
        }

    beam_params = _format_df_for_printing(
        beam_params,
        [["values", "errors"], ["alt_values", "alt_errors"]],
        units,
        new_varnames=varnames,
        latex=latex,
    )
    fit_params = _format_df_for_printing(
        fit_params, [["values", "errors"]], units, new_varnames=varnames, latex=latex
    )

    return fit_params, beam_params


def pretty_parameter_table(fit, beam, latex=False):
    # fit, beam = formatted_parameter_dfs(esme, latex=latex)

    tablefmt = "simple"
    if latex:
        tablefmt = "latex_raw"

    fit_table = tabulate.tabulate(
        fit, tablefmt=tablefmt, headers=["Variable", "Value", "Units"]
    )
    beam_table = tabulate.tabulate(
        beam, tablefmt=tablefmt, headers=["Variable", "Value", "Alt. Value", "Units"]
    )
    
    return f"{beam_table}\n\n\n{fit_table}"

# def control_room_table(fit, beam):




def _format_df_for_printing(
    df, value_error_name_pairs, units, new_varnames=None, latex=False
):
    if new_varnames is None:
        new_varnames = {}
    # Provide pairs of names of value column with associated error
    # column that should be combined into a single list of formatted
    # strings
    formatted_strings = {}
    trans = str.maketrans({"(": "", ")": ""})
    for value_col_name, error_col_name in value_error_name_pairs:
        formatted_strings[value_col_name] = []
        values, errors = df[value_col_name], df[error_col_name]
        for value, error in zip(values, errors):
            if error:
                pretty_value = f"{ufloat(value, error):.1u}"
            else:
                pretty_value = f"{value:.4g}"

            if latex:  # Typset for siunitx (latex)
                pm_symbol = "+-"
                pretty_value = pretty_value.translate(trans)
                pretty_value = rf"\num{{{pretty_value}}}"
            else:  # Typeset for just printing to terminal
                pm_symbol = "±"
            pretty_value = pretty_value.replace("+/-", pm_symbol)
            formatted_strings[value_col_name].append(pretty_value)

    # Add a units column to the df.
    var_names = df.index
    df_str = pd.DataFrame(formatted_strings, index=var_names)
    df_str["units"] = [units[name] for name in var_names]

    # Update the index, maybe to latex variables depending on new_varnames kwarg.
    new_index = []
    for name in df.index:
        try:
            new_index.append(new_varnames[name])
        except KeyError:
            new_index.append(name)
    df_str.index = new_index
    return df_str


def compare_results(esmes, latex=False):
    beam_dfs = []
    fit_dfs = []
    beam_units = []
    fit_units = []
    for i, esme in enumerate(esmes):
        beam, fit = formatted_parameter_dfs(esme, latex=latex)
        beam_units.append(beam["units"])
        fit_units.append(fit["units"])
        del fit["units"]
        del beam["units"]
        for key in beam:
            beam = beam.rename({key: f"{key.capitalize()} Dataset {i}"}, axis=1)
        for key in fit:
            fit = fit.rename({key: f"{key.capitalize()} Dataset {i}"}, axis=1)
        beam_dfs.append(beam)
        fit_dfs.append(fit)

    # Pick longest units column for consistency, and we only want a single units col.
    beam_units = beam_units[np.argmax([len(x) for x in beam_units])]
    fit_units = fit_units[np.argmax([len(x) for x in fit_units])]

    beam_comparision_df = pd.concat(beam_dfs, axis=1).fillna("-")
    beam_comparision_df["units"] = beam_units
    fit_comparision_df = pd.concat(fit_dfs, axis=1).fillna("-")
    fit_comparision_df["units"] = fit_units

    tablefmt = "simple"
    if latex:
        tablefmt = "latex_raw"

    fit_headers = ["Variable"] + list(fit_comparision_df.keys())
    beam_headers = ["Variable"] + list(beam_comparision_df.keys())
    fit_table = tabulate.tabulate(
        fit_comparision_df, tablefmt=tablefmt, headers=fit_headers
    )
    beam_table = tabulate.tabulate(
        beam_comparision_df, tablefmt=tablefmt, headers=beam_headers
    )

    return f"{fit_table}\n\n\n{beam_table}"


def format_latex_quantity():
    pass


def fexp(number):
    from decimal import Decimal

    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def plot_quad_strengths(
    esme, root_outdir=None
) -> plt.Figure:
    _plot_quad_strengths_dscan(esme, root_outdir=root_outdir)
    _plot_quad_strengths_tds(esme, root_outdir=root_outdir)
    if root_outdir is None:
        plt.show()


def _plot_quad_strengths(dfs, scan_var, scan_var_name, ax) -> plt.Figure:
    assert len(scan_var) == len(dfs)

    scan_quads = [lat.mean_quad_strengths(df) for df in dfs]

    for dx, df in zip(scan_var, scan_quads):
        ax.errorbar(
            df["s"],
            df["kick_mean"],
            yerr=df["kick_std"],
            label=f"{scan_var_name}={dx}",
            linestyle="",
            marker="x",
        )
    ax.legend()


def plot_tds_calibration(sesme, root_outdir):
    fig1, dikt1 = plot_calibrator_with_fits(sesme.dscan.calibrator)
    fig2, dikt2 = plot_r34s(sesme)
    fig3, dikt3 = plot_calibrated_tds(sesme)
    fig4, df4 = plot_streaking_parameters(sesme)
    fig5, dikt5 = plot_tds_voltage(sesme)

    if root_outdir is None:
        plt.show()
        return

    if fig1:
        fig1.savefig(root_outdir / "calibrator-fits.png")
    if fig2:
        path2 = root_outdir / "r34s.png"
        fig2.savefig(path2)
        with path2.with_suffix(".pcl").open("wb") as f:
            pickle.dump(dikt2, f)

    if fig3:
        path3 = root_outdir / "derived-tds-voltages.png"
        fig3.savefig(path3)
        with path3.with_suffix(".pcl").open("wb") as f:
            pickle.dump(dikt3, f)

    fig4.savefig(root_outdir / "streaking-parameters.png")

    path5 = root_outdir / "tds-calibration-slopes.png"
    fig5.savefig(path5)
    with path5.with_suffix(".pcl").open("wb") as f:
        pickle.dump(dikt5, f)


def plot_r34s(sesme):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))

    dscan_r34s = cal.r34s_from_scan(sesme.dscan)

    ax1.plot(sesme.dscan.dx, dscan_r34s, marker="x")
    ax1.set_xlabel(r"$\eta_\mathrm{{OTR}}\,/\,\mathrm{m}$")
    ax1.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")

    tscan_r34s = cal.r34s_from_scan(sesme.tscan)

    ax2.plot(sesme.tscan.tds_percentage, tscan_r34s, marker="x")
    ax2.set_xlabel(r"TDS Amplitude / %")
    ax2.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")

    dikt = {
        "dscan_dx": sesme.dscan.dx,
        "dscan_r34s": dscan_r34s,
        "tscan_percentage": sesme.tscan.tds_percentage,
        "tscan_r34s": tscan_r34s,
    }

    return fig, dikt


def plot_calibrator_with_fits(calib, fig=None, ax=None):
    if ax is None and fig is None:
        fig, ax = plt.subplots()

    x = calib.percentages
    sample_x = np.linspace(min(x) * 0.9, max(x) / 0.9, num=100)

    try:
        y = calib.tds_slopes * 1e-6
        calib.get_tds_slope(sample_x) * 1e-6
        ylabel = TDS_CALIBRATION_LABEL
    except AttributeError:  # Then it's a TrivialTDSCalibrator
        y = calib.voltages * 1e-6
        ylabel = VOLTAGE_LABEL
    else:
        ax.plot(sample_x, calib.get_tds_slope(sample_x) * 1e-6, label="Fit")
    ax.plot(x, y, label="Data")
    ax.set_xlabel(TDS_AMPLITUDE_LABEL)
    ax.set_ylabel(ylabel)

    ax.legend()
    df = None
    return fig, df


def plot_calibrated_tds(sesme):
    # We plot two things here.  Firstly the stuff Sergey wrote down yesterday.

    if isinstance(sesme.tscan.calibrator, cal.TrivialTDSCalibrator):
        return

    # Secondly the derived voltages for the TDS scan.

    tds_percentage = sesme.tscan.tds_percentage
    derived_voltage = _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme)

    sergey_percentages = sesme.tscan.calibrator.percentages

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))
    ax1.plot(
        tds_percentage, derived_voltage, marker="x", label="TDS Scan Derived Voltages"
    )

    sergey_tds_slopes = sesme.tscan.calibrator.tds_slopes
    popt, _ = sesme.tscan.calibrator.fit()

    ax2.plot(
        sergey_percentages,
        sergey_tds_slopes * 1e-6,
        marker="x",
        label="TDS Calibration data",
    )
    ax2.plot(
        sergey_percentages,
        maths.line(sergey_percentages, *popt) * 1e-6,
        marker="x",
        label="Fit",
    )

    ax2.set_ylabel(TDS_CALIBRATION_LABEL)

    ax2.set_xlabel("TDS Amplitude / %")
    ax1.set_ylabel(r"$|V_\mathrm{TDS}|$ / MV")
    ax2.legend()

    fig.suptitle(
        "TDS Calibration data we took beforehand (below) and applied to scan (above)"
    )

    dikt = {"tds_scan_percentages": tds_percentage, "tds_scan_voltage": derived_voltage}

    return fig, dikt


def _plot_scan_tds_rb_and_sp(scan, ax, colours, markers, label):
    all_rbs = scan.amplitude_rbs()
    # rb_means = rbs.mean(axis=1)
    # rb_stds = rbs.std(axis=1)
    sps = scan.amplitude_sps()
    # Setpoint is constant, but rbs will be different for every single image.
    # We plot for each setpoint mu
    colour = next(colours)
    marker = next(markers)
    for sp, rbs in zip(sps, all_rbs):
        print(sp)
        ax.scatter(np.ones_like(rbs) * sp, rbs, color=colour, marker=marker, label=label)
        label = None

def plot_amplitude_setpoints_with_readbacks(measurement, outdir):
    fig, ax = plt.subplots(figsize=(12.8, 8), sharey=True)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = iter(prop_cycle.by_key()['color'])
    markers = iter(["x", ".", "*"])

    _plot_scan_tds_rb_and_sp(measurement.dscan, ax, colours=colours, markers=markers, label="Dispersion Scan")
    _plot_scan_tds_rb_and_sp(measurement.tscan, ax, colours=colours, markers=markers, label="TDS Scan")
    _plot_scan_tds_rb_and_sp(measurement.bscan, ax, colours=colours, markers=markers, label="Beta Scan")

    ax.legend()
    ax.set_xlabel(TDS_AMPLITUDE_SETPOINT_LABEL)
    ax.set_ylabel(TDS_AMPLITUDE_READBACK_LABEL)

    return fig

def plot_tds_voltages(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iaxes = iter(axes)

    try:
        ax = next(iaxes)
        ax.plot(measurement.dscan.dispersions(), measurement.dscan.voltages()*1e-6)
        ax.set_xlabel(ETA_LABEL)
    except StopIteration:
        pass

    try:
        ax = next(iaxes)
        ax.plot(measurement.tscan.amplitude_sps(), measurement.tscan.voltages()*1e-6)
        ax.set_xlabel(TDS_AMPLITUDE_SETPOINT_LABEL)
    except StopIteration:
        pass

    try:
        ax = next(iaxes)
        ax.plot(measurement.bscan.betas(), measurement.bscan.voltages()*1e-6)
        ax.set_xlabel(BETA_LABEL)
    except StopIteration:
        pass

    axes[0].set_ylabel(VOLTAGE_LABEL)


    plt.show()
    return fig
    import sys
    sys.exit()


def _plot_scan_streaking_parameters(scan, axes, xvar, xlabel):
    s = _streaks_from_scan(scan)
    axes.plot(xvar, s)
    axes.set_xlabel(xlabel)


def _streaks_from_scan(scan):
    voltages = scan.voltages()
    energy = scan.beam_energies().mean() # this is in MeV
    r34s = _r34s_from_scan(scan)

    s = ana.streaking_parameter(voltage=voltages,
                                energy=energy*1e6, # Has to be eV
                                r12_streaking=r34s)
    return s


def _r34s_from_scan(scan):
    energy = scan.beam_energies().mean() # this is in MeV

    r34s = []
    for setpoint in scan.setpointdfs:
        r34 = optics.calculate_i1d_r34_from_tds_centre(setpoint.df,
                                                       "OTRC.64.I1D",
                                                       energy) # This has to be in MeV...
        r34s.append(r34)
    return np.array(r34s)

def plot_streaking_parameters(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_streaking_parameters(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_streaking_parameters(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_streaking_parameters(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    axes[0].set_ylabel(STREAKING_LABEL)


def plot_r12_streaking(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_r12_streaking(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_r12_streaking(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_r12_streaking(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    axes[0].set_ylabel(R34_LABEL)


def _plot_scan_r12_streaking(scan, axes, xvar, xlabel):
    r34s = _r34s_from_scan(scan)
    axes.plot(xvar, r34s)
    axes.set_xlabel(xlabel)


def _plot_scan_time_resolution(scan, axes, xvar, xlabel, scale=1):
    sfactors = abs(_streaks_from_scan(scan))
    beamsizes = _get_streaking_plane_beamsizes(scan)

    resolution = beamsizes / sfactors

    axes.plot(xvar, resolution * 1e6 * scale)
    axes.set_xlabel(xlabel)


def plot_slice_length(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_time_resolution(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_time_resolution(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_time_resolution(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    fig.suptitle("Longitudinal resolution (i.e. slice $\it{length}$ in bunch that contributes to signal at given point")

    axes[0].set_ylabel(LONG_RESOLUTION_LABEL)


def plot_apparent_bunch_lengths(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_apparent_bunch_lengths(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_apparent_bunch_lengths(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_apparent_bunch_lengths(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    axes[0].set_ylabel(r"Apparent Bunch Length / mm")

    plt.show()



def plot_true_bunch_lengths(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_true_bunch_lengths(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_true_bunch_lengths(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_true_bunch_lengths(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    axes[0].set_ylabel(r"True Bunch Length / mm")

    plt.show()


def _plot_scan_apparent_bunch_lengths(scan, axes, xvar, xlabel):
    values, errors = _get_scan_apparent_bunch_lengths(scan)

    axes.errorbar(xvar, values*1e3, yerr=errors*1e3)
    axes.set_xlabel(xlabel)

def _plot_scan_true_bunch_lengths(scan, axes, xvar, xlabel):
    values, errors = _get_scan_true_bunch_lengths(scan)

    axes.errorbar(xvar, values*1e3, yerr=errors*1e3)
    axes.set_xlabel(xlabel)


def _get_scan_true_bunch_lengths(scan):
    bunch_lengths = []
    errors = []
    for setpoint in scan.setpointdfs:
        bunch_length = ana.true_bunch_length_from_df(setpoint)
        bunch_lengths.append(bunch_length.n)
        errors.append(bunch_length.s)

    return np.array(bunch_lengths), np.array(errors)
    # from IPython import embed; embed()


    # axes.errorbar(xvar, values*1e3, yerr=errors*1e3)
    # axes.set_xlabel(xlabel)


def _get_scan_apparent_bunch_lengths(scan):
    mean_lengths = []
    for setpoint in scan.setpointdfs:
        lengths = []
        for raw_im, bg in setpoint.get_images_with_background():
            imp = image.filter_image(raw_im, bg=bg, crop=True)
            length, error = ana.apparent_bunch_length_from_processed_image(imp)
            lengths.append(ufloat(length, error))

        mean_lengths.append(np.mean(lengths))

    values = np.array([n.n for n in mean_lengths])
    errors = np.array([n.s for n in mean_lengths])

    return values, errors



def _plot_scan_apparent_rms_bunch_lengths(scan, axes, xvar, xlabel):
    pass

# def plot_normalised_slice_length(measurement, outdir)

# def plot_slice_e(measurement, outdir):
#     fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
#     iax = iter(axes)

#     scale = 1 / ana.PIXEL_SCALE_Y_UM
#     _plot_scan_time_resolution(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL, scale=scale)
#     _plot_scan_time_resolution(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL, scale=scale)
#     _plot_scan_time_resolution(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL, scale=scale)

#     fig.suptitle("Longitudinal resolution")

#     axes[0].set_ylabel(LONG_RESOLUTION_PIXELS_LABEL)


def plot_streaking_plane_beamsizes(measurement, outdir):
    fig, axes = plt.subplots(ncols=measurement.nfullscans(), figsize=(12.8, 8), sharey=True)
    iax = iter(axes)

    _plot_scan_streaking_plane_beamsizes(measurement.dscan, next(iax), measurement.dscan.dispersions(), xlabel=ETA_LABEL)
    _plot_scan_streaking_plane_beamsizes(measurement.tscan, next(iax), measurement.tscan.voltages() * 1e-6, xlabel=VOLTAGE_LABEL)
    _plot_scan_streaking_plane_beamsizes(measurement.bscan, next(iax), measurement.bscan.betas(), xlabel=BETA_LABEL)

    fig.suptitle("Initial beamsizes in the streaking plane (without any streaking).  Analytical (no data).")

    axes[0].set_ylabel(BEAM_SIZE_Y_LABEL)

    plt.show()

def _plot_scan_streaking_plane_beamsizes(scan, axes, xvar, xlabel):
    # OK!  So we don't have any measured beamsize data, so we assume
    # 0.4mmmrad, and we use the quadrupole strengths (whihc are
    # measured!) to get the optics at the screen.

    emittance = 0.4e-6
    sizes = _get_streaking_plane_beamsizes(scan, emittance=emittance)
    axes.plot(xvar, np.array(sizes) * 1e6)
    axes.set_xlabel(xlabel)


def _get_streaking_plane_beamsizes(scan, emittance=0.4e-6):
    emittance = 0.4e-6
    sizes = []
    for setpoint in scan.setpointdfs:
        df, _ = optics.optics_from_measurement_df(setpoint.df)
        beta_y = df[df.id == "OTRC.64.I1D"].beta_y.item()
        energy = df[df.id == "OTRC.64.I1D"].E.item()
        m_e_GeV = 0.511e-3
        gamma = energy / m_e_GeV
        geo_emittance = emittance / gamma

        beam_size = np.sqrt(beta_y * geo_emittance)
        sizes.append(beam_size)

    return sizes




def get_slice_core(pixels) -> tuple[npt.NDArray, npt.NDArray]:
    # Remove zeroes on either side of the slice and just get the
    # values where there is signal.
    nonzero_pixels = (pixels != 0).nonzero()[0]
    istart = nonzero_pixels.min()
    iend = nonzero_pixels.max()

    pixelcut = pixels[istart : iend + 1]
    pixel_index = np.arange(len(pixelcut))

    return pixel_index, pixelcut


def plot_tds_set_point_vs_readback(dscan_files, tscan_files, title=""):
    fname_amplitudes = []
    setpoint_amplitudes = []
    rb_amplitudes = []
    rb_stdevs = []
    for fname in tscan_files:
        ampl_fname = tds_magic_number_from_filename(fname)
        amp_rb = pd.read_pickle(fname)["XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"]
        amp = pd.read_pickle(fname)["XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"]

        fname_amplitudes.append(ampl_fname)
        setpoint_amplitudes.append(amp.mean())
        rb_amplitudes.append(amp_rb.mean())
        rb_stdevs.append(amp_rb.std())

    fig, ax1 = plt.subplots()
    sample = np.arange(0, 1.5 * max(fname_amplitudes))

    ax1.errorbar(fname_amplitudes, setpoint_amplitudes, label="Setpoint")
    ax1.errorbar(fname_amplitudes, rb_amplitudes, yerr=rb_stdevs, label="Readback")
    ax1.errorbar(sample, sample, label="$y=x$", linestyle="--")

    ax1.set_xlim(
        0.9 * min(fname_amplitudes), max(fname_amplitudes) + 0.1 * min(fname_amplitudes)
    )
    ax1.legend()
    ax1.set_xlabel("TDS setpoint from file name")
    ax1.set_ylabel("TDS Amplitude from DOOCs")
    ax1.set_title(title)

    plt.show()
