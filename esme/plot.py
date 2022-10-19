#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat

import esme.analysis as ana


def show_before_after_processing(measurement: ana.ScanMeasurement, index: int) -> None:
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
    fig.suptitle(
        fr"TDS No. = {self.tds}, $\eta_x={self.dx}\,\mathrm{{m}}$, image {index}, before/after image processing"
    )
    ax2.legend()
    ax4.legend()

    # ix1 is the "bottom" rather than the "top" because origin is in the top
    # left hand corner when plotting images!
    (ix0, ix1), (iy0, iy1) = ana.get_cropping_bounds(imp)
    ax1.set_ylim(ix1, ix0)
    ax1.set_xlim(iy0, iy1)

    plt.show()
    fig.savefig(f"{index}_check.png")


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

def plot_dispersion_scan(scan: ana.DispersionScan) -> tuple[ufloat, ufloat]:
    widths = np.asarray(list(scan.get_max_energy_slice_widths(padding=10)))
    dx = scan.dx

    x2, widths2, errors2 = ana.transform_variables_for_linear_fit(dx, widths)
    c, m = ana.linear_fit_to_pixel_stds(dx, widths)

    d2sample = np.linspace(0, 1.1 * max(dx**2))
    sigma2fit = ana.line(d2sample, m.n, c.n)

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

def plot_tds_scan(scan: ana.TDSScan) -> tuple[ufloat, ufloat]:
    widths = np.asarray(list(scan.get_max_energy_slice_widths(padding=10)))
    tds = scan.tds

     x2, widths2, errors2 = ana.transform_variables_for_linear_fit(tds, widths)
    c, m = ana.linear_fit_to_pixel_stds(tds, widths)

    tds2sample = np.linspace(0, 1.1 * max(tds**2))
    sigma2fit = ana.line(tds2sample, m.n, c.n)

    fig, ax = plt.subplots()
    ax.errorbar(tds**2, widths2, yerr=errors2, label="Data")
    ax.plot(tds2sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Power}^2\,/\,\mathrm{\%}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "\%^2", c, m)

    return c, m

# def _plot_scan(scan: ana.DispersionScan | ana.TDSScan):
#     pass

def add_info_box(ax, symbol, xunits, c, m):
    props = dict(boxstyle='round', facecolor='white',
                 alpha=0.5)

    textstr = '\n'.join([
        rf"$\sigma_M^2 = A_{{{symbol}}} +B_{{{symbol}}} {{{symbol}}}^2$",
        rf"$A_D = ({c.n:.2f}\pm{c.s:.1g})\,\mathrm{{\mu {xunits}}}$",
        rf"$B_D = ({m.n:.2f}\pm{m.s:.1g})\,\mathrm{{\mu m^2\,/\,{xunits}}}$"])

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
