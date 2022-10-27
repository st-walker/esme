#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Union

import tabulate
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

def plot_dispersion_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> tuple[ufloat, ufloat]:
    scan = esme.dscan
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    dx2 = scan.dx**2

    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")
    a0, a1 = ana.linear_fit(dx2, widths_um2, errors_um2)

    d2sample = np.linspace(0, 1.1 * max(dx2))
    sigma2fit = ana.line(d2sample, a0.n, a1.n)

    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(dx2, widths_um2, yerr=errors_um2, label="Data")
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


def plot_tds_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> tuple[ufloat, ufloat]:
    widths, errors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    voltages = esme.oconfig.tds_voltages

    voltages2_mv2 = (voltages * 1e-6) ** 2
    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")

    a0, a1 = ana.linear_fit(voltages2_mv2, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    v2_sample = np.linspace(0.9 * min(voltages2_mv2), 1.1 * max(voltages2_mv2))
    sigma2fit = ana.line(v2_sample, a0.n, a1.n)

    if ax is None:
        fig, ax = plt.subplots()
    ax.errorbar(voltages2_mv2, widths_um2, yerr=errors_um2, label="Data")
    ax.plot(v2_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Voltage}^2\,/\,\mathrm{MV}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "MV", a0, a1)


def plot_scans(esme: ana.SliceEnergySpreadMeasurement):
    fig, (ax1, ax2) = plt.subplots(ncols=2)

    plot_dispersion_scan(esme, ax1)
    plot_tds_scan(esme, ax2)

    fig.suptitle("Dispersion and TDS Scan for a Energy Spread Measurement")


def add_info_box(ax, symbol, xunits, c, m):
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    textstr = '\n'.join(
        [
            rf"$\sigma_M^2 = A_{{{symbol}}} +B_{{{symbol}}} {{{symbol}}}^2$",
            rf"$A_{{{symbol}}} = ({c.n:.2f}\pm{c.s:.1g})\,\mathrm{{\mu m}}^2$",
            rf"$B_{{{symbol}}} = ({m.n:.2f}\pm{m.s:.1g})\,\mathrm{{\mu m^2\,/\,{xunits}}}^2$",
        ]
    )

    ax.text(0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top', bbox=props)


def plot_measured_central_widths(esme: ana.SliceEnergySpreadMeasurement):
    fig, ax = plt.subplots()

    dscan = esme.dscan
    tscan = esme.tscan
    voltages = esme.oconfig.voltages

    dx = dscan.dx

    dwidths = np.asarray(list(dscan.max_energy_slice_widths_and_errors(padding=10)))
    twidths = np.asarray(list(tscan.max_energy_slice_widths_and_errors(padding=10)))

    dwidths_um = dwidths * ana.PIXEL_SCALE_X_UM
    twidths_um = twidths * ana.PIXEL_SCALE_X_UM

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    ax1.errorbar(dx, dwidths[..., 0], yerr=dwidths[..., 1], marker="x")
    ax3.errorbar(dx, dwidths_um[..., 0], yerr=dwidths_um[..., 1], marker="x")
    ax2.errorbar(voltages, twidths[..., 0], yerr=twidths[..., 1], marker="x")
    ax4.errorbar(voltages, twidths_um[..., 0], yerr=twidths_um[..., 1], marker="x")

    ax1.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax3.set_ylabel(r"$\sigma_M\,/\,\mathrm{\mu m}$")
    ax3.set_xlabel("D / m")
    ax4.set_xlabel("TDS Voltage / V")

    fig.suptitle(fr"Measured maximum-energy slice widths for pixel scale Y = {ana.PIXEL_SCALE_X_UM} $\mathrm{{\mu m}}$")


def pretty_beam_parameter_table(esme: ana.SliceEnergySpreadMeasurement) -> str:
    params = esme.all_fit_parameters()
    from IPython import embed; embed()
    av, bv = params.a_v, params.b_v
    ad, bd = params.a_d, params.b_d

    sige, sige_err = params.sigma_e
    sige *= 1e-3
    sige_err *= 1e-3

    ex, exe = params.emitx
    ex *= 1e6
    exe *= 1e6

    header = ["Variable", "value", "error", "units"]
    variables = ["A_V", "B_V", "A_D", "B_D", "σ_E", "σ_I", "σ_B", "σ_R", "εₙ"]
    with_errors = [av, bv, ad, bd, (sige, sige_err), params.sigma_i,
                   params.sigma_b, params.sigma_r, (ex, exe)]

    units = ["m²", "m²/V²", "m²", "-", "keV", "m", "m", "m", "mm⋅mrad"]
    values = [a[0] for a in with_errors]
    errors = [a[1] for a in with_errors]

    return tabulate.tabulate(np.array([variables, values, errors, units]).T,
                             headers=header)




def pretty_measured_beam_sizes(esme: ana.SliceEnergySpreadMeasurement) -> str:
    raise ValueError
    params = esme.all_fit_parameters()
    from IPython import embed; embed()
