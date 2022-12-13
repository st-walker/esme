#!/usr/bin/env python3

import logging

import latdraw
import matplotlib.pyplot as plt
import numpy as np
import tabulate
from scipy.constants import e, c
from uncertainties import ufloat
import pandas as pd


import esme.analysis as ana
import esme.calibration as cal
import esme.lattice as lat

LOG = logging.getLogger(__name__)

ETA_LABEL = r"$\eta_\mathrm{{OTR}}\,/\,\mathrm{m}$"
VOLTAGE_LABEL = r"$|V_\mathrm{TDS}|\,/\,\mathrm{MV}$"
TDS_CALIBRATION_LABEL = r"TDS Calibration Gradient / $\mathrm{\mu{}mps^{-1}}$"
TDS_AMPLITUDE_LABEL = r"TDS Amplitude / %"


def dump_full_scan(esme: ana.SliceEnergySpreadMeasurement, root_outdir) -> None:
    dispersion_scan = esme.dscan
    tds_scan = esme.tscan

    dscan_dir = root_outdir / "dispersion-scan"
    for i, measurement in enumerate(dispersion_scan):
        dx = measurement.dx
        # tds = dispersion_scan.tds_percentage

        LOG.debug(f"starting to plot before/after for dscan, {dx=}m")

        measurement_outdir = dscan_dir / f"{i=},{dx=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)

        for image_index in range(measurement.nimages):
            LOG.debug(f"plotting before/after for image number: {image_index}")
            fig = show_before_after_processing(measurement, image_index)

            if root_outdir is not None:
                fig.savefig(measurement_outdir / f"{image_index}.png")
                plt.close()
            else:
                plt.show()

    dscan_dir = root_outdir / "tds-scan"
    for i, measurement in enumerate(tds_scan):
        # dx = measurement.dx
        tds = measurement.tds_percentage

        measurement_outdir = dscan_dir / f"{i=},{tds=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"starting to plot before/after for tds scan, tds = {tds}%")

        for image_index in range(measurement.nimages):
            LOG.debug(f"plotting before/after for image number: {image_index}")
            fig = show_before_after_processing(measurement, image_index)
            if root_outdir:
                fig.savefig(measurement_outdir / f"{image_index}.png")
                plt.close()
            else:
                plt.show()

    bscan = esme.bscan
    if not bscan:
        return
    bscan_dir = root_outdir / "beta-scan"
    for i, measurement in enumerate(tds_scan):
        # dx = measurement.dx
        beta = measurement.beta

        measurement_outdir = bscan_dir / f"{i=},{beta=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"starting to plot before/after for beta scan, beta = {beta}%")

        for image_index in range(measurement.nimages):
            LOG.debug(f"plotting before/after for image number: {image_index}")
            fig = show_before_after_processing(measurement, image_index)
            if root_outdir:
                fig.savefig(measurement_outdir / f"{image_index}.png")
                plt.close()
            else:
                plt.show()


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
    fig.suptitle(
        fr"TDS No. = {m.tds_percentage}, $\eta_\mathrm{{OTR}}={m.dx}\,\mathrm{{m}}$, image {index}, before/after image processing"
    )
    ax2.legend()
    ax4.legend()

    # ix1 is the "bottom" rather than the "top" because origin is in the top
    # left hand corner when plotting images!
    (ix0, ix1), (iy0, iy1) = ana.get_cropping_bounds(imp)
    ax1.set_ylim(ix1, ix0)
    ax1.set_xlim(iy0, iy1)

    return fig


def plot_dispersion_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    scan = esme.dscan
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    dx2 = scan.dx**2

    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um", to_variances=True)
    a0, a1 = ana.linear_fit(dx2, widths_um2, errors_um2)

    d2sample = np.linspace(0, 1.1 * max(dx2))
    sigma2fit = ana.line(d2sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(dx2, widths_um2, yerr=errors_um2, label="Data", marker=".", linestyle="")
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


def plot_tds_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    widths, errors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    voltages = esme.tscan.tds_voltage

    voltages2_mv2 = (voltages * 1e-6) ** 2
    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")

    a0, a1 = ana.linear_fit(voltages2_mv2, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    v2_sample = np.linspace(0.9 * min(voltages2_mv2), 1.1 * max(voltages2_mv2))
    sigma2fit = ana.line(v2_sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(voltages2_mv2, widths_um2, yerr=errors_um2, label="Data", marker=".", linestyle="")
    ax.plot(v2_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Voltage}^2\,/\,\mathrm{MV}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "MV", a0, a1)


def plot_beta_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    widths, errors = esme.bscan.max_energy_slice_widths_and_errors(padding=10)

    beta = esme.bscan.beta

    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")

    a0, a1 = ana.linear_fit(beta, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    beta_sample = np.linspace(0.9 * min(beta), 1.1 * max(beta))
    sigma2fit = ana.line(beta_sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.errorbar(beta, widths_um2, yerr=errors_um2, label="Data", marker=".", linestyle="")
    ax.plot(beta_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\beta_\mathrm{OCR}\,/\,\mathrm{m}$")
    ax.set_title("Beta scan fit")
    add_info_box(ax, r"\beta", "m", a0, a1)




def plot_scans(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> None:
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

    ax2.set_ylabel("")
    ax3.set_ylabel("")

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

    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)


def plot_measured_central_widths(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None, show=True, write_widths=True) -> None:
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    dscan = esme.dscan
    tscan = esme.tscan
    voltages = esme.tscan.tds_percentage

    dx = dscan.dx

    if dscan.measurements:
        dwidths, derrors = dscan.max_energy_slice_widths_and_errors(padding=10)
        dwidths_um, derrors_um = ana.transform_pixel_widths(dwidths, derrors, pixel_units="um", to_variances=False)
        plot_scan_central_widths(dscan, dx, ax1, ax3)

        if write_widths:
            data = {"dx": dx, "px_widths": dwidths, "px_errors": derrors, "um_widths": dwidths_um, "um_errors": derrors_um}
            pd.DataFrame.from_dict(data).to_pickle(hardcoded / "dscan-pixels.pcl")


    if tscan.measurements:
        twidths, terrors = tscan.max_energy_slice_widths_and_errors(padding=10)
        twidths_um, terrors_um = ana.transform_pixel_widths(twidths, terrors, pixel_units="um", to_variances=False)
        plot_scan_central_widths(tscan, voltages, ax2, ax4)

        if write_widths:
            data = {"voltage": voltages, "px_widths": twidths, "px_errors": terrors, "um_widths": twidths_um, "um_errors": terrors_um}
            pd.DataFrame.from_dict(data).to_pickle(hardcoded / "tscan-pixels.pcl")


    ax1.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax3.set_ylabel(r"$\sigma_M\,/\,\mathrm{\mu m}$")
    ax3.set_xlabel("D / m")
    # ax4.set_xlabel("TDS Voltage / MV")
    ax4.set_xlabel("TDS Amplitude / %")

    fig.suptitle(fr"Measured maximum-energy slice widths for pixel scale X = {ana.PIXEL_SCALE_X_UM} $\mathrm{{\mu m}}$")

    if show:
        plt.show()

    if root_outdir is not None:
        fig.savefig(root_outdir / "measured-central-widths.png")


def plot_scan_central_widths(scan: ana.DispersionScan, x, ax1, ax2):
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    widths_um, errors_um = ana.transform_pixel_widths(widths, errors, pixel_units="um", to_variances=False)

    ax1.errorbar(x, widths, yerr=errors, marker="x")
    ax2.errorbar(x, widths_um, yerr=errors_um, marker="x")


# def _um_tuple(pair):
#     return pair*-

def compare_both_derivations():
    pass

def pretty_beam_parameter_table(esme: ana.SliceEnergySpreadMeasurement, latex=False) -> str:
    params = esme.all_fit_parameters()

    av, bv = params.a_v, params.b_v
    ad, bd = params.a_d, params.b_d

    sige, sige_err = params.sigma_e
    sige *= 1e-3
    sige_err *= 1e-3

    ex, exe = params.emitx
    ex *= 1e6
    exe *= 1e6

    reference_voltage = params.reference_voltage
    reference_dispersion = params.reference_dispersion
    
    with_errors = [(reference_voltage*1e-6, 0),
                   (reference_dispersion, 0),
                   av,
                   bv,
                   ad,
                   bd,
                   (np.nan, np.nan), # a_beta
                   (np.nan, np.nan), # b_beta
                   (sige, sige_err),
                   params.sigma_i,
                   params.sigma_b,
                   params.sigma_r,
                   (ex, exe)
                   ]

    variables_with_units = [("V_0", "MV"),
                            ("D_0", "m"),
                            ("A_V", "m²"),
                            ("B_V", "m²/V²"),
                            ("A_D", "m²"),
                            ("B_D", "-"),
                            ("A_beta", "m²"),
                            ("B_beta", "m"),
                            ("σ_E", "keV"),
                            ("σ_I", "m"),
                            ("σ_B", "m"),
                            ("σ_R", "m"),
                            ("emitx", "mm⋅mrad")]

    latex_variables_with_units = [(r"$V_0$", r"\si{\mega\volt}"),
                                  (r"$D_0$", r"\si{\metre}"),
                                  (r"$A_V$", r"\si{\metre\squared}"),
                                  (r"$B_V$", r"\si{\metre\squared\per\volt\squared}"),
                                  (r"$A_D$", r"\si{\metre\squared}"),
                                  (r"$B_D$", ""),
                                  (r"$A_\beta$", r"\si{\metre\squared}"),
                                  (r"$B_\beta$", r"\si{\metre}"),
                                  (r"$\sigma_E$", r"$\kilo\electron\volt$"),
                                  (r"$\sigma_I$", r"\si{\metre}"),
                                  (r"$\sigma_B$", r"\si{\metre}"),
                                  (r"$\sigma_R$", r"\si{\metre}"),
                                  (r"$\varepsilon_x$", r"\si{\milli\metre{}\cdot{}\milli\radian}")
                                  ]

    exa, exae = params.emitx_alt
    exa *= 1e6
    exae *= 1e6
    sige_alt, sige_alt_err = params.sigma_e_alt
    sige_alt *= 1e-3
    sige_alt_err *= 1e-3
    with_errors_alt = [(reference_voltage*1e-6, 0),
                       (reference_dispersion, 0),
                       av,
                       bv,
                       ad,
                       bd,
                       (np.nan, np.nan),
                       (np.nan, np.nan),
                       (sige_alt, sige_alt_err),
                       params.sigma_i_alt,
                       ]

    if esme.bscan:
        with_errors_alt[-4] = params.a_beta
        with_errors_alt[-3] = params.b_beta

        with_errors_alt.extend([params.sigma_b_alt,
                                params.sigma_r_alt,
                                (exa, exae)
                                ])
    else:
        missing_number = (np.nan, np.nan)
        with_errors_alt.extend(5 * [missing_number])

        # beta_scan_variables = [(r"$A_beta$", "\si{\metre\squared}"),
        #                        (r"$B_beta$", "\si{\metre\squared}"),
        #                        (r"$sigma_b_alt$", "\si{\metre}"), # Needs a_beta
        #                        (r"$sigma_r_alt$", "\si{\metre}"), # Needs a_beta
        #                        (r"$emitx_alt$", r"\si{\milli\metre{}\cdot{}\milli\radian}")]
        # latex_beta_scan_variables = [(r"$A_\beta$", r"\si{\metre\squared}"),
        #                              (r"$B_\beta$", r"\si{\metre\squared}"),
        #                              (r"\sigma_b_alt", r"\si{\metre}"), # Needs a_beta
        #                              (r"\sigma_r_{\beta}", r"\si{\metre}"), # Needs a_beta
        #                              (r"\varepsilon_{\beta}", r"\si{\milli\metre{}\cdot{}\milli\radian}"),
        #                              (r"$V_0$", r"\si{\mega\volt}"),
        #                              (r"$D_0$", r"\si{\metre}")]

        alt_variables_with_units.append(beta_scan_variables)
        # latex_alt_variables_with_units.append(latex_beta_scan_variables)


    if latex:
        variables_with_units = latex_variables_with_units
        alt_variables_with_units = latex_alt_variables_with_units

    variables = [v[0] for v in variables_with_units]
    units = [v[1] for v in variables_with_units]


    formatted_strings = []

    for value, error in with_errors:
        formatted_strings.append(f"{ufloat(value, error):.1u}")
    formatted_strings_alt = []
    for value, error in with_errors_alt:
        formatted_strings_alt.append(f"{ufloat(value, error):.1u}")

    header = ["Variable", "Value", "Alt. Value", "Units"]

    from IPython import embed; embed()
    
    return header, variables, formatted_strings, formatted_strings_alt, units
    # return tabulate.tabulate(np.array([variables, formatted_strings, units]).T, headers=header, tablefmt=tablefmt)

def compare_results(esmes, latex=False):
    rows = []
    for esme in esmes:
        header, variables, values, units = pretty_beam_parameter_table(esme, False)
        rows.append(values)

    tablefmt = "simple"
    if latex:
        tablefmt = "latex_raw"
    if len(rows) == 1:
        header = ["Variable", "value"]
    else:
        header = ["Variable"] + [f"sim {i}" for (i, _) in enumerate(esmes)]

    pd.DataFrame.from_dict({title: row for title, row in zip(header, rows)})
    tab = tabulate.tabulate(np.array([variables, units, *rows]).T,
                             headers=header,
                             tablefmt=tablefmt)
    return tab

# import numpy as np
# def decompose_float(x: np.float32):
#     """decomposes a float32 into negative, exponent, and significand"""
#     negative = x < 0
#     n = np.abs(x).view(np.int32) # discard sign (MSB now 0),
#                                  # view bit string as int32
#     exponent = (n >> 23) - 127 # drop significand, correct exponent offset
#                                # 23 and 127 are specific to float32
#     significand = n & np.int32(2**23 - 1) # second factor provides mask
#                                           # to extract significand
#     return (negative, exponent, significand)


def fexp(number):
    from decimal import Decimal
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1


def plot_quad_strengths(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> plt.Figure:
    _plot_quad_strengths_dscan(esme, root_outdir=root_outdir)
    _plot_quad_strengths_tds(esme, root_outdir=root_outdir)
    if root_outdir is None:
        plt.show()


def _plot_quad_strengths_dscan(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> plt.Figure:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))
    ((ax1, ax2), (ax3, ax4)) = axes
    axes = axes.flatten()

    dscan_all_images_dfs = esme.dscan.metadata
    scan_quads = [lat.mean_quad_strengths(df) for df in dscan_all_images_dfs]

    design_dispersions, design_quads = lat.design_quad_strengths()

    for dx_actual, dx_design, df_actual, df_design, ax in zip(
        design_dispersions, esme.dscan.dx, design_quads, scan_quads, axes
    ):
        ax.set_title(rf"Dispersion scan optics for $D_x={dx_actual}\,\mathrm{{m}}$")
        ax.errorbar(
            df_actual.s, df_actual.kick_mean, yerr=df_actual.kick_std, label="Readback", linestyle="", marker="x"
        )
        ax.plot(df_design.s, df_design.kick_mean, linestyle="", label="Intended", marker=".")
        # ax.bar(df_design.s, df_design.kick_mean, alpha=0.25)

    ylabel = r"$K_1 L\,/\,\mathrm{mrad\cdot{}m^{-1}}$"
    ax1.set_ylabel(ylabel)
    ax3.set_ylabel(ylabel)
    ax3.set_xticks(list(df_design.s), list(df_design.index), rotation=60, fontsize=8)
    ax4.set_xticks(list(df_design.s), list(df_design.index), rotation=60, fontsize=8)
    ax3.set_xlabel("Quadrupole name")
    ax4.set_xlabel("Quadrupole name")

    axes[0].legend()
    fig.suptitle("Dispersion scan quadrupole strengths, 2021 TDS Calibration")

    if root_outdir is not None:
        fig.savefig(root_outdir / "dscan-quads.png")


def _plot_quad_strengths_tds(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None):
    # fig, ax = plt.subplots()

    cell = lat.injector_cell()
    fig, (axm, ax) = latdraw.subplots_with_lattices(
        [latdraw.interfaces.lattice_from_ocelot(cell), None], figsize=(14, 8)
    )

    tscan_all_images_dfs = esme.tscan.metadata
    tscan_dx = esme.tscan.dx[0]
    assert (tscan_dx == esme.tscan.dx).all()

    tds_scan_quads = [lat.mean_quad_strengths(df) for df in tscan_all_images_dfs]
    voltages = esme.tscan.tds_voltage / 1e6  # to MV
    for voltage, df_actual in zip(voltages, tds_scan_quads):
        ax.errorbar(
            df_actual.s,
            df_actual.kick_mean,
            yerr=df_actual.kick_std,
            label=f"V={voltage:.3g} MV",
            linestyle="",
            marker="x",
        )

    design_dispersions, design_quads = lat.design_quad_strengths()
    index = np.abs(np.array(design_dispersions) - tscan_dx).argmin()
    df = design_quads[index]
    ax.plot(df.s, df.kick_mean, linestyle="", label="Intended", marker=".")
    ax.legend()
    axm.set_title("TDS Scan quadrupole strengths, 2021 TDS Calibration")

    ax.set_xticks(list(df.s), list(df.index), rotation=60, fontsize=8)
    ax.set_ylabel(r"$K_1 L\,/\,\mathrm{mrad\cdot{}m^{-1}}$")

    if root_outdir is not None:
        fig.savefig(root_outdir / "tscan-quads.png")


def _plot_quad_strengths(dfs, scan_var, scan_var_name, ax) -> plt.Figure:
    assert len(scan_var) == len(dfs)

    scan_quads = [lat.mean_quad_strengths(df) for df in dfs]

    for dx, df in zip(scan_var, scan_quads):
        ax.errorbar(
            df["s"], df["kick_mean"], yerr=df["kick_std"], label=f"{scan_var_name}={dx}", linestyle="", marker="x"
        )
    ax.legend()


def plot_tds_calibration(sesme, root_outdir):
    fig1 = plot_calibrator_with_fits(sesme.dscan.calibrator)
    fig2 = plot_r34s(sesme)
    fig3 = plot_calibrated_tds(sesme)
    fig4 = plot_streaking_parameters(sesme)
    fig5 = plot_tds_voltage(sesme)
    fig6 = plot_bunch_lengths(sesme)

    if root_outdir is None:
        plt.show()
        return

    fig1.savefig(root_outdir / "calibrator-fits.png")
    fig2.savefig(root_outdir / "r34s.png")
    fig3.savefig(root_outdir / "derived-tds-voltages.png")
    fig4.savefig(root_outdir / "streaking-parameters.png")
    fig5.savefig(root_outdir / "tds-calibration-slopes.png")
    fig6.savefig(root_outdir / "bunch-lengths.png")


def _r34s_from_scan(scan: ana.ParameterScan):
    result = []
    for measurement in scan:
        # Pick a non-bg image.
        im = measurement.images[0]
        result.append(cal.r34_from_tds_to_screen(im.metadata))
    return np.array(result)


def plot_r34s(sesme):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))

    dscan_r34s = _r34s_from_scan(sesme.dscan)

    ax1.plot(sesme.dscan.dx, dscan_r34s, marker="x")
    ax1.set_xlabel(r"$\eta_\mathrm{{OTR}}\,/\,\mathrm{m}$")
    ax1.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")

    tscan_r34s = _r34s_from_scan(sesme.tscan)

    ax2.plot(sesme.tscan.tds_percentage, tscan_r34s, marker="x")
    ax2.set_xlabel(r"TDS Amplitude / %")
    ax2.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")

    return fig

def plot_calibrator_with_fits(calib):
    fig, ax = plt.subplots()

    sample_x = np.linspace(min(calib.percentages) * 0.9,
                           max(calib.percentages) / 0.9,
                           num=100)

    ax.plot(calib.percentages, calib.tds_slopes * 1e-6, marker="x", linestyle="", label="Data")
    calib.fn = cal.line
    ax.plot(sample_x, calib.get_tds_slope(sample_x) * 1e-6, label="line(%)")
    calib.fn = cal.sqrt
    ax.plot(sample_x, calib.get_tds_slope(sample_x) * 1e-6, label="sqrt(%)")
    # Change it back =/..  this is obviously quite error prone...
    calib.fn = cal.line

    ax.legend()
    ax.set_xlabel(TDS_AMPLITUDE_LABEL)
    ax.set_ylabel(TDS_CALIBRATION_LABEL)
    return fig


def plot_calibrated_tds(sesme):
    # We plot two things here.  Firstly the stuff Sergey wrote down yesterday.

    # Secondly the derived voltages for the TDS scan.

    # What we actually used in our scan:
    tds_percentage = sesme.tscan.tds_percentage
    derived_tds_voltage = abs(sesme.tscan.tds_voltage * 1e-6)  # MV

    sergey_percentages = sesme.tscan.calibrator.percentages
    # sergey_voltages = abs(
    #     sesme.tscan.calibrator.get_voltage(sergey_percentages, sesme.tscan[0].images[0].metadata) * 1e-6
    # )

    # popt, _ = curve_fit(ana.line, sergey_percentages, sergey_sergey_)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))
    ax1.plot(tds_percentage, derived_tds_voltage, marker="x", label="TDS Scan Derived Voltages")

    sergey_tds_slopes = sesme.tscan.calibrator.tds_slopes
    popt, _ = sesme.tscan.calibrator.fit()

    ax2.plot(sergey_percentages, sergey_tds_slopes * 1e-6, marker="x", label="TDS Calibration data")
    ax2.plot(sergey_percentages, ana.line(sergey_percentages, *popt) * 1e-6,
             marker="x", label="Fit")

    ax2.set_ylabel(TDS_CALIBRATION_LABEL)

    ax2.set_xlabel("TDS Amplitude / %")
    ax1.set_ylabel(r"$|V_\mathrm{TDS}|$ / MV")
    ax2.legend()

    fig.suptitle("TDS Calibration data we took beforehand (below) and applied to scan (above)")

    return fig


def _streaks_from_scan(scan: ana.ParameterScan):
    scan_voltages = scan.tds_voltage
    energy = scan.beam_energy() * e # in eV and convert to joules

    k0 = e * abs(scan_voltages) * cal.TDS_WAVENUMBER / energy
    r34s = _r34s_from_scan(scan)
    streak = r34s * k0

    return streak


def plot_tds_voltage(sesme):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.8, 8), sharey=True)

    dscan = sesme.dscan
    dscan_dx = dscan.dx
    dscan_voltage = abs(dscan.tds_voltage * 1e-6)

    ax1.plot(dscan_dx, dscan_voltage, marker="x")
    ax1.set_ylabel(VOLTAGE_LABEL)
    ax1.set_xlabel(ETA_LABEL)
    ax2.set_xlabel(r"TDS Amplitude / %")

    tscan = sesme.tscan
    tscan_percent = tscan.tds_percentage
    tscan_voltage = abs(tscan.tds_voltage * 1e-6)

    ax2.plot(tscan_percent, tscan_voltage, marker="x")
    ax2.set_xlabel(r"TDS Amplitude / %")

    # Should have all the same percentages for the dispersion scan.
    dscan_pc = sesme.dscan.tds_percentage
    assert (dscan_pc == dscan_pc[0]).all()
    # Should have same dispersion throughout for the tds scan.
    tscan_dx = sesme.tscan.dx
    assert (tscan_dx == tscan_dx[0]).all()

    tit1 = rf"$\eta_\mathrm{{OTR}}$-scan; TDS Amplitude at ${{{dscan_pc[0]}}}\%$"
    ax1.set_title(tit1)
    tit2 = rf"TDS-scan; $\eta_\mathrm{{OTR}}={{{tscan_dx[0]}}}\mathrm{{m}}$"
    ax2.set_title(tit2)

    return fig


def plot_streaking_parameters(sesme):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 8), sharey=True)

    dscan_streak = abs(_streaks_from_scan(sesme.dscan))

    ax1.plot(sesme.dscan.dx, dscan_streak)

    ax1.set_xlabel(ETA_LABEL)
    ax1.set_ylabel(r"$|S|$")

    ax2.set_xlabel(VOLTAGE_LABEL)

    tscan_streak = abs(_streaks_from_scan(sesme.tscan))

    ax2.plot(abs(sesme.tscan.tds_voltage*1e-6), tscan_streak)
    return fig

def plot_bunch_lengths(sesme):
    dscan = sesme.dscan
    zrms = [m.zrms(pixel_units="m") for m in dscan]

    zrms = np.array(zrms)
    zrmsn = zrms[..., 0]
    zrmse = zrms[..., 1]


    dscan_streak = abs(_streaks_from_scan(dscan))
    dscan_bunch_length = zrms


    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.errorbar(dscan.dx, 1e12 * zrmsn / dscan_streak / c,
                 yerr=1e12*zrmse / dscan_streak / c)
    ax1.set_xlabel(ETA_LABEL)
    ax1.set_ylabel(r"$\sigma_z\,/\,\mathrm{ps}$")

    tscan = sesme.tscan
    zrms = [m.zrms(pixel_units="m") for m in tscan]

    zrms = np.array(zrms)
    zrmsn = zrms[..., 0]
    zrmse = zrms[..., 1]

    tscan_streak = abs(_streaks_from_scan(tscan))
    tscan_bunch_length = zrms


    ax2.errorbar(1e-6*abs(tscan.tds_voltage), 1e12 * zrmsn / tscan_streak / c,
                 yerr=1e12*zrmse / tscan_streak / c)
    ax2.set_xlabel(VOLTAGE_LABEL)

    return fig
