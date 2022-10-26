#!/usr/bin/env python3

import os
from typing import Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

import esme.analysis as ana

# def all_measurements_before_after_processing(measurement: ana.ScanMeasurement) -> None:
#     for meas

def dump_full_scan(dispersion_scan, tds_scan, root_outdir):

    dscan_dir = root_outdir / "dispersion-scan"
    for i, measurement in enumerate(dispersion_scan):
        dx = measurement.dx
        # tds = dispersion_scan.tds

        measurement_outdir = dscan_dir / f"{i=},{dx=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)

        for image_index in range(measurement.nimages):
            fig = show_before_after_processing(measurement, image_index)
            fig.savefig(measurement_outdir / f"{image_index}.png")
            plt.close()



    dscan_dir = root_outdir / "tds-scan"
    for i, measurement in enumerate(tds_scan):
        # dx = measurement.dx
        tds = measurement.tds

        measurement_outdir = dscan_dir / f"{i=},{tds=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)

        for image_index in range(measurement.nimages):
            fig = show_before_after_processing(measurement, image_index)
            fig.savefig(measurement_outdir / f"{image_index}.png")
            plt.close()



def show_before_after_processing(measurement: ana.ScanMeasurement, index: int) -> plt.Figure:
    im = measurement.to_im(index, process=False)
    imp = measurement.to_im(index, process=True)

    fig = plt.figure(figsize=(14, 8))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(2, 2, 4, sharex=ax2, sharey=ax2)

    ax1.imshow(im, aspect="auto")
    ax3.imshow(imp, aspect="auto")

    x, slice_mus, _ = ana.get_slice_properties(imp)
    idx_emax = x[np.argmin(slice_mus)]  # Min not max because image index counts from top.

    ax1.axvline(idx_emax, alpha=0.25, color="white")
    ax3.axvline(idx_emax, alpha=0.25, color="white")

    padding = 10
    slc = im[..., idx_emax - padding : idx_emax + padding].mean(axis=1)
    slcp = imp[..., idx_emax - padding : idx_emax + padding].mean(axis=1)

    bg = measurement.mean_bg_im()[..., idx_emax - padding : idx_emax + padding].mean(axis=1)

    ax2.plot(slc, label="Raw")
    ax4.plot(slcp, label="After processing")
    ax2.plot(bg, label="Background")
    ax2.plot((slc - bg).clip(min=0), label="Raw - background")

    shift = np.argmax(slcp)
    xcore, ycore = ana.get_slice_core(slcp)
    popt, perr = ana.get_gaussian_fit(xcore, ycore)

    popt[1] = shift  # Shift mean back to whatever it was.
    vertical_index = np.linspace(popt[1] - 50, popt[1] + 50, 100)
    y = ana.gauss(vertical_index, *popt)
    ax4.plot(vertical_index, y, label="Fit")

    ax4.set_xlim(min(vertical_index), max(vertical_index))

    sigma = popt[-1]
    sigma_sigma = perr[-1]

    ax4.set_title(rf"Fitted $\sigma_M = {sigma:.3f}±{sigma_sigma:.3f}$ px")

    # Left plots
    ax1.set_title("Image before and after processing")
    ax3.set_xlabel("Pixel column index")
    ax1.set_ylabel("Pixel row index")
    ax3.set_ylabel("Pixel row index")
    # Right plots
    ax2.set_title("Highest energy column")
    ax4.set_xlabel("Pixel Row Index")
    ax2.set_ylabel("Pixel Brightness")
    ax4.set_ylabel("Pixel Brightness")
    m = measurement
    fig.suptitle(fr"TDS No. = {m.tds}, $\eta_x={m.dx}\,\mathrm{{m}}$, image {index}, before/after image processing")
    ax2.legend()
    ax4.legend()

    # ix1 is the "bottom" rather than the "top" because origin is in the top
    # left hand corner when plotting images!
    (ix0, ix1), (iy0, iy1) = ana.get_cropping_bounds(imp)
    ax1.set_ylim(ix1, ix0)
    ax1.set_xlim(iy0, iy1)


    return fig
    # plt.show()
    # fig.savefig(f"{index}_check.png")


# def show_before_after_for_measurement(self, index: int) -> None:
#     for i in range(self.measurements[index].nimages):
#         self.measurements[index].show_before_after_processing(i)

# def show_image(image: RawImageT) -> None:
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     plt.show()


# def plot_image(image: RawImageT) -> None:
#     fig, ax = plt.subplots()
#     ax.imshow(image)
#     return fig, ax

# def plot_dispersion_scan(gradient, y_intercept, dispersion, widths):

def plot_dispersion_scan(scan: ana.DispersionScan, ax=None) -> tuple[ufloat, ufloat]:
    widths = np.asarray(list(scan.get_max_energy_slice_widths(padding=10)))
    dx = scan.dx

    x2, widths2, errors2 = ana.transform_variables_for_linear_fit(dx, widths)
    c, m = ana.linear_fit_to_pixel_stds(dx, widths)

    d2sample = np.linspace(0, 1.1 * max(dx**2))
    sigma2fit = ana.line(d2sample, m.n, c.n)

    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(dx**2, widths2, yerr=errors2, label="Data")
    ax.plot(d2sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$D^2\,/\,\mathrm{m}^2$")
    ax.set_title("Dispersion scan fit")
    add_info_box(ax, "D", "m^2", c, m)

    return c, m


def _set_ylabel_for_scan(ax):
    ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{\mathrm{\mu}m}^2$")
    # ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{m}^2$")
    # ax.set_ylabel(r"$\sigma_M^2\,/\,\mathrm{px}^2$")


def plot_tds_scan(scan: ana.TDSScan, ax=None) -> tuple[ufloat, ufloat]:
    widths = np.asarray(list(scan.get_max_energy_slice_widths(padding=10)))
    tds = scan.tds

    x2, widths2, errors2 = ana.transform_variables_for_linear_fit(tds, widths)
    c, m = ana.linear_fit_to_pixel_stds(tds, widths)

    tds2sample = np.linspace(0, 1.1 * max(tds**2))
    sigma2fit = ana.line(tds2sample, m.n, c.n)

    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(tds**2, widths2, yerr=errors2, label="Data")
    ax.plot(tds2sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Power}^2\,/\,\mathrm{\%}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "\\%^2", c, m)

    return c, m

def plot_scans(scan1, scan2):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    plot_dispersion_scan(scan1, ax1)
    plot_tds_scan(scan2, ax2)

    fig.suptitle("Dispersion and TDS Scan for a Energy Spread Measurement")



def add_info_box(ax, symbol, xunits, c, m):
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    textstr = '\n'.join(
        [
            rf"$\sigma_M^2 = A_{{{symbol}}} +B_{{{symbol}}} {{{symbol}}}^2$",
            rf"$A_D = ({c.n:.2f}\pm{c.s:.1g})\,\mathrm{{\mu {xunits}}}$",
            rf"$B_D = ({m.n:.2f}\pm{m.s:.1g})\,\mathrm{{\mu m^2\,/\,{xunits}}}$",
        ]
    )

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)


def plot_measured_central_widths(dscan:ana.DispersionScan, tscan: ana.TDSScan):
    fig, ax = plt.subplots()

    dx = dscan.dx
    # tds = [0.38,0.47,0.56,0.65,0.75] # ??

    tds = tscan.tds

    dwidths = np.asarray(list(dscan.get_max_energy_slice_widths(padding=10)))
    twidths = np.asarray(list(tscan.get_max_energy_slice_widths(padding=10)))

    dwidths_um = dwidths * ana.PIXEL_SIZE_X_UM
    twidths_um = twidths * ana.PIXEL_SIZE_X_UM

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.errorbar(dx, dwidths[..., 0], yerr=dwidths[..., 1], marker="x")
    ax3.errorbar(dx, dwidths_um[..., 0],
                 yerr=dwidths_um[..., 1], marker="x")
    ax2.errorbar(tds, twidths[..., 0], yerr=twidths[..., 1], marker="x")
    ax4.errorbar(tds, twidths_um[..., 0],
                 yerr=twidths_um[..., 1], marker="x")

    ax1.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax3.set_ylabel(r"$\sigma_M\,/\,\mathrm{\mu m}$")
    ax3.set_xlabel("D / m")
    ax4.set_xlabel("TDS strength / %")

    fig.suptitle(fr"Measured maximum-energy slice widths for pixel size Y = {ana.PIXEL_SIZE_Y_UM} $\mathrm{{\mu m}}$")
