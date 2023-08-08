"""Set of functions for plotting and make results tables"""


import logging
import pickle
import shutil

import matplotlib.pyplot as plt
import numpy as np
import tabulate
from scipy.constants import e, c
from uncertainties import ufloat
import pandas as pd
import numpy.typing as npt


import esme.analysis as ana
import esme.calibration as cal
import esme.lattice as lat
import esme.beam as beam
import esme.maths as maths
import esme.image as image
from esme.exceptions import TDSCalibrationError

LOG = logging.getLogger(__name__)

SLICE_WIDTH_LABEL = r"$\sigma_M\,/\,\mathrm{\mu m}$"
ETA_LABEL = r"$\eta_\mathrm{{OTR}}\,/\,\mathrm{m}$"
ETA2_LABEL = r"$\eta^2_\mathrm{{OTR}}\,/\,\mathrm{m}$"
VOLTAGE_LABEL = r"$|V_\mathrm{TDS}|\,/\,\mathrm{MV}$"
VOLTAGE2_LABEL = r"$|V^2_\mathrm{TDS}|\,/\,\mathrm{MV}$"
TDS_CALIBRATION_LABEL = r"Gradient / $\mathrm{\mu{}mps^{-1}}$"
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

        data_dir = measurement_outdir / "raw-beam-images"
        data_dir.mkdir(exist_ok=True)

        for image_index in range(measurement.nimages):
            LOG.debug(f"plotting before/after for image number: {image_index}")
            fig = show_before_after_processing(measurement, image_index)

            image_file_path = measurement[image_index].filename

            shutil.copy(image_file_path, data_dir / image_file_path.name)

            if root_outdir is not None:
                fig.savefig(measurement_outdir / f"{image_index}.png")
                plt.close()
            else:
                plt.show()

        background_dir = measurement_outdir / "raw-background-images"
        background_dir.mkdir(exist_ok=True)
        for image_index, image in enumerate(measurement.bg):
            image_file_path = image.filename
            shutil.copy(image_file_path, background_dir / image_file_path.name)



    dscan_dir = root_outdir / "tds-scan"
    for i, measurement in enumerate(tds_scan):
        # dx = measurement.dx
        tds = measurement.tds_percentage

        measurement_outdir = dscan_dir / f"{i=},{tds=}"
        measurement_outdir.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"starting to plot before/after for tds scan, tds = {tds}%")

        data_dir = measurement_outdir / "raw-beam-images"
        data_dir.mkdir(exist_ok=True)


        for image_index in range(measurement.nimages):
            LOG.debug(f"plotting before/after for image number: {image_index}")
            fig = show_before_after_processing(measurement, image_index)
            if root_outdir:
                fig.savefig(measurement_outdir / f"{image_index}.png")
                plt.close()
            else:
                plt.show()

            image_file_path = measurement[image_index].filename
            save = shutil.copy(image_file_path, data_dir / image_file_path.name)


        background_dir = measurement_outdir / "raw-background-images"
        background_dir.mkdir(exist_ok=True)
        for image_index, image in enumerate(measurement.bg):
            image_file_path = image.filename
            shutil.copy(image_file_path, background_dir / image_file_path.name)


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
    xcore, ycore = get_slice_core(slcp)
    popt, perr = maths.get_gaussian_fit(xcore, ycore)

    popt[1] = shift  # Shift mean back to whatever it was.
    vertical_index = np.linspace(popt[1] - 50, popt[1] + 50, 100)
    y = maths.gauss(vertical_index, *popt)
    ax4.plot(vertical_index, y, label="Fit")

    ax4.set_xlim(min(vertical_index), max(vertical_index))

    sigma = popt[-1]
    sigma_sigma = perr[-1]

    ax4.set_title(rf"Fitted $\sigma_M = {sigma:.3f}±{sigma_sigma:.3f}$ px")

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
    fname = measurement.metadata["XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"].iloc[index].name
    fig.suptitle(
        fr"TDS Ampl. = {m.tds_percentage}%, $D_\mathrm{{OTR}}={m.dx}\,\mathrm{{m}}$, image {index}, before/after image processing: {fname}"
    )
    ax2.legend()
    ax4.legend()

    # ix1 is the "bottom" rather than the "top" because origin is in the top
    # left hand corner when plotting images!
    (ix0, ix1), (iy0, iy1) = image.get_cropping_bounds(imp)
    ax1.set_ylim(ix1, ix0)
    ax1.set_xlim(iy0, iy1)

    return fig


def plot_dispersion_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    scan = esme.dscan
    widths, errors = scan.max_energy_slice_widths_and_errors(padding=10)
    dx2 = scan.dx**2

    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um", to_variances=True)
    a0, a1 = maths.linear_fit(dx2, widths_um2, errors_um2)

    d2sample = np.linspace(0, 1.1 * max(dx2))
    sigma2fit = maths.line(d2sample, a0[0], a1[0])

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


def write_pixel_widths(esme, outdir):
    dwidths, derrors = esme.dscan.max_energy_slice_widths_and_errors(padding=10)
    twidths, terrors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    dwidths_um, derrors_um = ana.transform_pixel_widths(dwidths, derrors, pixel_units="um", to_variances=False)
    twidths_um, terrors_um = ana.transform_pixel_widths(twidths, terrors, pixel_units="um", to_variances=False)
    dscan_data = pd.DataFrame({
        "dx": esme.dscan.dx,
        "amplitude": esme.dscan.tds_percentage,
        "voltage": esme.dscan.voltage,
        "px_widths": dwidths,
        "px_errors": derrors,
        "um_widths": dwidths_um,
        "um_errors": derrors_um,
    })

    tds_data = pd.DataFrame({
        "dx": esme.tscan.dx,
        "amplitude": esme.tscan.tds_percentage,
        "voltage": esme.tscan.voltage,
        "px_widths": twidths,
        "px_errors": terrors,
        "um_widths": twidths_um,
        "um_errors": terrors_um,
    })

    dscan_data.to_csv(outdir / "dscan_central_slices.csv")
    tds_data.to_csv(outdir / "tscan_central_slices.csv")    



def plot_tds_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    widths, errors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    voltages_mv = _get_tds_tscan_abs_voltage_in_mv_from_scans(esme)

    voltages2_mv2 = voltages_mv ** 2
    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")

    a0, a1 = maths.linear_fit(voltages2_mv2, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    v2_sample = np.linspace(0.9 * min(voltages2_mv2), 1.1 * max(voltages2_mv2))
    sigma2fit = maths.line(v2_sample, a0[0], a1[0])

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

    a0, a1 = maths.linear_fit(beta, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    beta_sample = np.linspace(0.9 * min(beta), 1.1 * max(beta))
    sigma2fit = maths.line(beta_sample, a0[0], a1[0])

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


def plot_measured_central_widths(
    esme: ana.SliceEnergySpreadMeasurement, root_outdir=None, show=True, write_widths=True
) -> None:
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
        twidths_um, terrors_um = ana.transform_pixel_widths(twidths, terrors, pixel_units="um", to_variances=False)
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

    fit_params = params.fit_parameters_to_df()
    fit_params.loc["sigma_z"] = beam.mean_bunch_length_from_time_calibration(esme)
    fit_params.loc["sigma_z"] *= 1e3  # to mm
    fit_params.loc["sigma_t"] = fit_params.loc["sigma_z"] * 1e-3 * 1e12 / c  # to ps
    fit_params.loc[["V_0", "E_0"]] *= 1e-6  # To MV / MeV

    beam_params = params.beam_parameters_to_df()
    beam_params.loc[["emitx", "sigma_i", "sigma_b", "sigma_r"]] *= 1e6  # to mm.mrad & um
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
    from IPython import embed; embed()

    if latex:
        varnames = latex_variables
        units = {var_name: latex_units[unit_str] for (var_name, unit_str) in units.items()}

    beam_params = _format_df_for_printing(
        beam_params, [["values", "errors"], ["alt_values", "alt_errors"]], units, new_varnames=varnames, latex=latex
    )
    fit_params = _format_df_for_printing(fit_params, [["values", "errors"]], units, new_varnames=varnames, latex=latex)

    return fit_params, beam_params


def pretty_parameter_table(fit, beam, latex=False):
    # fit, beam = formatted_parameter_dfs(esme, latex=latex)

    tablefmt = "simple"
    if latex:
        tablefmt = "latex_raw"

    fit_table = tabulate.tabulate(fit, tablefmt=tablefmt, headers=["Variable", "Value", "Units"])
    beam_table = tabulate.tabulate(beam, tablefmt=tablefmt, headers=["Variable", "Value", "Alt. Value", "Units"])

    return f"{beam_table}\n\n\n{fit_table}"


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
    import latdraw  # pylint: disable=import-error

    cell = lat.cell_to_injector_dump()
    fig, (axm, ax) = latdraw.subplots_with_lattices(
        [latdraw.interfaces.lattice_from_ocelot(cell), None], figsize=(14, 8)
    )

    tscan_all_images_dfs = esme.tscan.metadata
    tscan_dx = esme.tscan.dx[0]
    assert (tscan_dx == esme.tscan.dx).all()

    tds_scan_quads = [lat.mean_quad_strengths(df) for df in tscan_all_images_dfs]
    voltages = _get_tds_tscan_abs_voltage_in_mv_from_scans(esme)
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

    dikt = {"dscan_dx": sesme.dscan.dx, "dscan_r34s": dscan_r34s, "tscan_percentage": sesme.tscan.tds_percentage, "tscan_r34s": tscan_r34s}

    return fig, dikt


def plot_calibrator_with_fits(calib, fig=None, ax=None):
    if ax is None and fig is None:
        fig, ax = plt.subplots()

    x = calib.percentages
    sample_x = np.linspace(min(x) * 0.9, max(x) / 0.9, num=100)

    try:
        y = calib.tds_slopes * 1e-6
        yfit = calib.get_tds_slope(sample_x) * 1e-6
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
    ax1.plot(tds_percentage, derived_voltage, marker="x", label="TDS Scan Derived Voltages")

    sergey_tds_slopes = sesme.tscan.calibrator.tds_slopes
    popt, _ = sesme.tscan.calibrator.fit()

    ax2.plot(sergey_percentages, sergey_tds_slopes * 1e-6, marker="x", label="TDS Calibration data")
    ax2.plot(sergey_percentages, maths.line(sergey_percentages, *popt) * 1e-6, marker="x", label="Fit")

    ax2.set_ylabel(TDS_CALIBRATION_LABEL)

    ax2.set_xlabel("TDS Amplitude / %")
    ax1.set_ylabel(r"$|V_\mathrm{TDS}|$ / MV")
    ax2.legend()

    fig.suptitle("TDS Calibration data we took beforehand (below) and applied to scan (above)")

    dikt = {"tds_scan_percentages": tds_percentage, "tds_scan_voltage": derived_voltage}

    return fig, dikt


def _streaks_from_scan(scan: ana.ParameterScan, scan_voltages=None):
    if scan_voltages is None:
        scan_voltages = scan.voltage
    energy = scan.beam_energy() * e  # in eV and convert to joules
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
    tscan_voltage = _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme)

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

    dikt = {"dscan_dx": dscan_dx, "dscan_voltage": dscan_voltage, "tscan_percent": tscan_percent, "tscan_voltage": tscan_voltage}

    return fig, dikt


def plot_streaking_parameters(sesme):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=(14, 8), sharey=False)

    dscan_streak = abs(_streaks_from_scan(sesme.dscan))

    raw_bunch_lengths, raw_bl_errors = beam.apparent_bunch_lengths(sesme.dscan)
    true_bunch_lengths, true_bl_errors = beam.true_bunch_lengths(sesme.dscan)

    # To ps
    def to_ps(length):
        return (length / c) * 1e12

    ax3.errorbar(
        sesme.dscan.dx, to_ps(raw_bunch_lengths), linestyle="", yerr=to_ps(raw_bl_errors), label="Raw bunch length", marker=".",
    )

    ax3.errorbar(
        sesme.dscan.dx, to_ps(true_bunch_lengths), linestyle="", yerr=to_ps(true_bl_errors), label="True bunch length", marker=".",
    )

    ax3.legend()
    # ax3.plot(sesme.dscan.dx, image_lengths, label="True bunch length")

    ax1.plot(sesme.dscan.dx, dscan_streak, marker="x", linestyle="")


    ax1.set_ylabel(r"$|S|$")

    tscan_streak = abs(_streaks_from_scan(
        sesme.tscan,
        scan_voltages=_get_tds_tscan_abs_voltage_in_mv_from_scans(sesme) * 1e6)
                       )

    ax2.plot(_get_tds_tscan_abs_voltage_in_mv_from_scans(sesme), tscan_streak, marker="x", linestyle="")

    raw_bunch_lengths, raw_bl_errors = beam.apparent_bunch_lengths(sesme.tscan)
    true_bunch_lengths, true_bl_errors = beam.true_bunch_lengths(sesme.tscan, _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme)*1e6)

    ax4.errorbar(
        _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme),
        to_ps(raw_bunch_lengths),
        linestyle="",
        marker=".",
        yerr=to_ps(raw_bl_errors),
        label="Raw bunch length",
    )

    ax4.errorbar(
        _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme),
        to_ps(true_bunch_lengths),
        linestyle="",
        marker=".",
        yerr=to_ps(true_bl_errors),
        label="True bunch length",
    )

    ax3.set_ylabel("Bunch Length / ps")
    ax3.set_xlabel(ETA_LABEL)
    ax4.set_xlabel(VOLTAGE_LABEL)

    df = None

    return fig, df


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

    ax1.set_xlim(0.9 * min(fname_amplitudes), max(fname_amplitudes) + 0.1 * min(fname_amplitudes))
    ax1.legend()
    ax1.set_xlabel("TDS setpoint from file name")
    ax1.set_ylabel("TDS Amplitude from DOOCs")
    ax1.set_title(title)

    plt.show()


def _get_tds_tscan_abs_voltage_in_mv_from_scans(sesme):
    """this is simply to handle the case where i accidentally
    calibrated the TDS at a different dispersion to what i did the
    tds scan at"""
    # What we actually used in our scan:
    tds_percentage = sesme.tscan.tds_percentage
    try:
        derived_voltage = abs(sesme.tscan.voltage * 1e-6)  # MV
    except TDSCalibrationError:
        # Then I guess I accidentally calibrated the TDS at the wrong
        # dispersion sp.  oops!  Calculate it more "by hand" by
        # getting the correct snapshot (and therefore R34) from the
        # *dispersion* scan, and then use that to calculate the TDS voltage.
        correct_snapshot = np.array(sesme.dscan.measurements)[sesme.dscan.dx == sesme.dscan.calibrator.dispersion_setpoint].item().metadata.iloc[0]
        derived_voltage = abs(sesme.tscan.calibrator.get_voltage(tds_percentage, correct_snapshot)) * 1e-6
    return derived_voltage
