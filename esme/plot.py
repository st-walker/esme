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
import esme.beam as beam

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
    for i, measurement in enumerate(bscan):
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

    y, slice_mus, _ = ana.get_slice_properties(imp)
    idy_emax = y[np.argmin(slice_mus)]  # Min not max because image index counts from top.

    ax1.axhline(idy_emax, alpha=0.25, color="white")
    ax3.axhline(idy_emax, alpha=0.25, color="white")

    padding = 10
    central_slice_index = np.s_[idy_emax - padding : idy_emax + padding]
    slc = im[central_slice_index].mean(axis=0)
    slcp = imp[central_slice_index].mean(axis=0)

    bg = measurement.mean_bg_im()[central_slice_index].mean(axis=0)

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

    ax4.set_title(rf"Fitted $\sigma_M = {sigma:.3f}??{sigma_sigma:.3f}$ px")

    # Left plots
    ax1.set_title("Image before and after processing")
    ax3.set_xlabel("Pixel Column index")
    ax1.set_ylabel("Pixel Row index")
    ax3.set_ylabel("Pixel Row index")
    # Right plots
    ax2.set_title("Highest energy column")
    ax4.set_xlabel("Pixel Column Index")
    ax2.set_ylabel("Pixel Brightness")
    ax4.set_ylabel("Pixel Brightness")
    m = measurement
    fig.suptitle(
        fr"TDS No. = {m.tds_percentage}, D_\mathrm{{OTR}}={m.dx}\,\mathrm{{m}}$, image {index}, before/after image processing"
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
    voltages = esme.tscan.voltage

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

def compare_both_derivations(test, latex=True):
    pass


# def _derived_paramaters(params):

def _coefficients_df(params):
    pass

def formatted_parameter_dfs(esme: ana.SliceEnergySpreadMeasurement, latex=False) -> str:
    params = esme.all_fit_parameters()



    beam_params = params.beam_parameters_to_df()
    fit_params = params.fit_parameters_to_df()

    fit_params.loc[["V_0", "E_0"]] *= 1e-6 # To MV / MeV
    beam_params.loc[["emitx", "sigma_i", "sigma_b", "sigma_r"]] *= 1e6 # to mm.mrad & um
    beam_params.loc["sigma_e"] *= 1e-3 # to keV

    units = {'V_0': 'MV',
             'D_0': 'm',
             'E_0': 'MeV',
             'A_V': 'm2',
             'B_V': 'm2/V2',
             'A_D': 'm2',
             'B_D': '',
             'A_beta': 'm2',
             'B_beta': 'm',
             'sigma_e': 'keV',
             'sigma_i': 'um',
             'sigma_b': 'um',
             'sigma_r': 'um',
             'emitx': 'mm.mrad'}

    latex_variables = {'V_0': '$V_0$',
                       'D_0': '$D_0$',
                       'E_0': '$E_0$',
                       'A_V': '$A_V$',
                       'B_V': '$B_V$',
                       'A_D': '$A_D$',
                       'B_D': '$B_D$',
                       'A_beta': '$A_\\beta$',
                       'B_beta': '$B_\\beta$',
                       'sigma_e': '$\\sigma_E$',
                       'sigma_i': '$\\sigma_I$',
                       'sigma_b': '$\\sigma_B$',
                       'sigma_r': '$\\sigma_R$',
                       'emitx': '$\\varepsilon_x$'}

    latex_units = {"m2": r"\si{\metre\squared}",
                   "MV": r"\si{\mega\volt}",
                   "m": r"\si{\metre}",
                   "um": r"\si{\micro\metre}",
                   "m2/V2": r"\si{\metre\squared\per\volt\squared}",
                   "keV": r"\si{\kilo\electronvolt}",
                   "MeV": r"\si{\mega\electronvolt}",
                   "": "",
                   "mm.mrad": r"\si{\milli\metre{}\cdot{}\milli\radian}"}

    varnames = None
    if latex:
        varnames = latex_variables
        units = {var_name: latex_units[unit_str] for (var_name, unit_str) in units.items()}

    beam_params = _format_df_for_printing(beam_params, [["values", "errors"], ["alt_values", "alt_errors"]], units, new_varnames=varnames, latex=latex)
    fit_params = _format_df_for_printing(fit_params, [["values", "errors"]], units, new_varnames=varnames, latex=latex)

    return fit_params, beam_params


def pretty_parameter_table(esme, latex=False):
    fit, beam = formatted_parameter_dfs(esme, latex=latex)

    tablefmt = "simple"
    if latex:
        tablefmt = "latex_raw"

    fit_table = tabulate.tabulate(fit, tablefmt=tablefmt, headers=["Variable", "Value", "Units"])
    beam_table = tabulate.tabulate(beam, tablefmt=tablefmt, headers=["Variable", "Value", "Alt. Value", "Units"])

    return f"{fit_table}\n\n\n{beam_table}"

def _format_df_for_printing(df, value_error_name_pairs, units, new_varnames=None, latex=False):
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
            pretty_value = f"{ufloat(value, error):.1u}"
            if latex: # Typset for siunitx (latex)
                pm_symbol = "+-"
                pretty_value = pretty_value.translate(trans)
                pretty_value = rf"\num{{{pretty_value}}}"
            else: # Typeset for just printing to terminal
                pm_symbol = "??"
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
    fit_table = tabulate.tabulate(fit_comparision_df, tablefmt=tablefmt, headers=fit_headers)
    beam_table = tabulate.tabulate(beam_comparision_df, tablefmt=tablefmt, headers=beam_headers)

    return f"{fit_table}\n\n\n{beam_table}"



def format_latex_quantity():
    pass


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
    voltages = esme.tscan.voltage / 1e6  # to MV
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
    derived_voltage = abs(sesme.tscan.voltage * 1e-6)  # MV

    sergey_percentages = sesme.tscan.calibrator.percentages
    # sergey_voltages = abs(
    #     sesme.tscan.calibrator.get_voltage(sergey_percentages, sesme.tscan[0].images[0].metadata) * 1e-6
    # )

    # popt, _ = curve_fit(ana.line, sergey_percentages, sergey_sergey_)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 8))
    ax1.plot(tds_percentage, derived_voltage, marker="x", label="TDS Scan Derived Voltages")

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
    scan_voltages = scan.voltage
    energy = scan.beam_energy() * e # in eV and convert to joules
    k0 = e * abs(scan_voltages) * cal.TDS_WAVENUMBER / energy
    r34s = cal.r34s_from_scan(scan)
    streak = r34s * k0

    return streak

def plot_tds_voltage(sesme):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12.8, 8), sharey=True)

    dscan = sesme.dscan
    dscan_dx = dscan.dx
    dscan_voltage = abs(dscan.voltage * 1e-6)

    ax1.plot(dscan_dx, dscan_voltage, marker="x")
    ax1.set_ylabel(VOLTAGE_LABEL)
    ax1.set_xlabel(ETA_LABEL)
    ax2.set_xlabel(r"TDS Amplitude / %")

    tscan = sesme.tscan
    tscan_percent = tscan.tds_percentage
    tscan_voltage = abs(tscan.voltage * 1e-6)

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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(14, 8), sharey=False)

    dscan_streak = abs(_streaks_from_scan(sesme.dscan))

    raw_bunch_lengths, raw_bl_errors = beam.apparent_bunch_lengths(sesme.dscan)
    true_bunch_lengths, true_bl_errors = beam.true_bunch_lengths(sesme.dscan)

    # To ps
    def to_ps(length):
        return (length / c) * 1e12

    # raw_bl_ps = to_ps(raw_bunch_lengths)
    # raw_bl_ps_errors = to_ps(raw_bl_errors)

    # true_bl_ps = to_ps(true_bunch_lengths)
    # true_bl_ps_errors = to_ps(true_bl_errors)

    ax3.errorbar(sesme.dscan.dx,
                 to_ps(raw_bunch_lengths),
                 yerr=to_ps(raw_bl_errors),
                 label="Raw bunch length")

    ax3.errorbar(sesme.dscan.dx,
                 to_ps(true_bunch_lengths),
                 yerr=to_ps(true_bl_errors),
                 label="true bunch length")

    ax3.legend()
    # ax3.plot(sesme.dscan.dx, image_lengths, label="True bunch length")

    ax1.plot(sesme.dscan.dx, dscan_streak)

    ax1.set_xlabel(ETA_LABEL)
    ax1.set_ylabel(r"$|S|$")

    ax2.set_xlabel(VOLTAGE_LABEL)

    tscan_streak = abs(_streaks_from_scan(sesme.tscan))

    ax2.plot(abs(sesme.tscan.voltage*1e-6), tscan_streak)

    raw_bunch_lengths, raw_bl_errors = beam.apparent_bunch_lengths(sesme.tscan)
    true_bunch_lengths, true_bl_errors = beam.true_bunch_lengths(sesme.tscan)


    ax4.errorbar(sesme.tscan.voltage * 1e-6,
                 to_ps(raw_bunch_lengths),
                 yerr=to_ps(raw_bl_errors),
                 label="Raw bunch length")

    ax4.errorbar(sesme.tscan.voltage * 1e-6,
                 to_ps(true_bunch_lengths),
                 yerr=to_ps(true_bl_errors),
                 label="True bunch length")

    ax3.set_ylabel("Bunch Length / ps")
    ax3.set_xlabel("Dx")
    ax4.set_xlabel("TDS Voltage / MV")

    plt.show()


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


    ax2.errorbar(1e-6*abs(tscan.voltage), 1e12 * zrmsn / tscan_streak / c,
                 yerr=1e12*zrmse / tscan_streak / c)
    ax2.set_xlabel(VOLTAGE_LABEL)

    return fig
