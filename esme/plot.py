#!/usr/bin/env python3

import latdraw
import matplotlib.pyplot as plt
import numpy as np
import tabulate

import esme.analysis as ana
import esme.calibration as cal
import esme.lattice as lat


def dump_full_scan(esme: ana.SliceEnergySpreadMeasurement, root_outdir) -> None:

    dispersion_scan = esme.dscan
    tds_scan = esme.tscan

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


def plot_tds_scan(esme: ana.SliceEnergySpreadMeasurement, ax=None) -> None:
    widths, errors = esme.tscan.max_energy_slice_widths_and_errors(padding=10)
    voltages = esme.oconfig.tds_voltages

    voltages2_mv2 = (voltages * 1e-6) ** 2
    widths_um2, errors_um2 = ana.transform_pixel_widths(widths, errors, pixel_units="um")

    a0, a1 = ana.linear_fit(voltages2_mv2, widths_um2, errors_um2)

    # Sample from just below minimum voltage to just above maximum
    v2_sample = np.linspace(0.9 * min(voltages2_mv2), 1.1 * max(voltages2_mv2))
    sigma2fit = ana.line(v2_sample, a0[0], a1[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    ax.errorbar(voltages2_mv2, widths_um2, yerr=errors_um2, label="Data")
    ax.plot(v2_sample, sigma2fit, label="Fit")
    ax.legend(loc="lower right")

    _set_ylabel_for_scan(ax)

    ax.set_xlabel(r"$\mathrm{TDS\ Voltage}^2\,/\,\mathrm{MV}^2$")
    ax.set_title("TDS scan fit")
    add_info_box(ax, "V", "MV", a0, a1)


def plot_scans(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> None:
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 8))

    plot_dispersion_scan(esme, ax1)
    plot_tds_scan(esme, ax2)

    fig.suptitle("Dispersion and TDS Scan for a Energy Spread Measurement")

    if root_outdir is not None:
        fig.savefig(root_outdir / "scan-fits.png")


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


def plot_measured_central_widths(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))

    dscan = esme.dscan
    tscan = esme.tscan
    voltages = esme.oconfig.tds_voltages

    dx = dscan.dx

    dwidths, derrors = dscan.max_energy_slice_widths_and_errors(padding=10)
    twidths, terrors = tscan.max_energy_slice_widths_and_errors(padding=10)

    dwidths_um, derrors_um = ana.transform_pixel_widths(dwidths, derrors, pixel_units="um", to_variances=False)
    twidths_um, terrors_um = ana.transform_pixel_widths(twidths, terrors, pixel_units="um", to_variances=False)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    ax1.errorbar(dx, dwidths, yerr=derrors, marker="x")
    ax3.errorbar(dx, dwidths_um, yerr=derrors_um, marker="x")

    ax2.errorbar(voltages * 1e-6, twidths, yerr=terrors, marker="x")
    ax4.errorbar(voltages * 1e-6, twidths_um, yerr=terrors_um, marker="x")

    ax1.set_ylabel(r"$\sigma_M\,/\,\mathrm{px}$")
    ax3.set_ylabel(r"$\sigma_M\,/\,\mathrm{\mu m}$")
    ax3.set_xlabel("D / m")
    ax4.set_xlabel("TDS Voltage / MV")

    fig.suptitle(fr"Measured maximum-energy slice widths for pixel scale Y = {ana.PIXEL_SCALE_X_UM} $\mathrm{{\mu m}}$")

    if root_outdir is not None:
        fig.savefig(root_outdir / "measured-central-widths.png")


def pretty_beam_parameter_table(esme: ana.SliceEnergySpreadMeasurement) -> str:
    params = esme.all_fit_parameters()

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
    with_errors = [av, bv, ad, bd, (sige, sige_err), params.sigma_i, params.sigma_b, params.sigma_r, (ex, exe)]

    units = ["m²", "m²/V²", "m²", "-", "keV", "m", "m", "m", "mm⋅mrad"]
    values = [a[0] for a in with_errors]
    errors = [a[1] for a in with_errors]

    return tabulate.tabulate(np.array([variables, values, errors, units]).T, headers=header)


def plot_quad_strengths(esme: ana.SliceEnergySpreadMeasurement, root_outdir=None) -> plt.Figure:
    _plot_quad_strengths_dscan(esme, root_outdir=root_outdir)
    _plot_quad_strengths_tds(esme, root_outdir=root_outdir)


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
    voltages = esme.oconfig.tds_voltages / 1e6  # to MV
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
    plot_r34s(sesme, root_outdir)
    plot_tds_slopes(sesme, root_outdir)
    plot_tds_voltages(sesme, root_outdir)


def _r34s_from_scan(scan: ana.TDSDispersionScan):
    result = []
    for measurement in scan:
        # Pick a non-bg image.
        im = measurement.images[0]
        result.append(cal.r34_from_tds_to_screen(im.metadata))
    return result


def plot_r34s(sesme, root_outdir):
    fig, (ax1, ax2) = plt.subplots(nrows=2)

    dscan_r34s = _r34s_from_scan(sesme.dscan)

    ax1.plot(sesme.dscan.dx, dscan_r34s)
    ax1.set_xlabel(r"$\eta_x\,/\,\mathrm{m}$")
    ax1.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")

    tscan_r34s = _r34s_from_scan(sesme.tscan)

    ax2.plot(sesme.tscan.tds, tscan_r34s)
    ax2.set_xlabel(r'TDS "Power" / %')
    ax2.set_ylabel(r"$R_{34}\,/\,\mathrm{m\cdot{}rad^{-1}}$")


def plot_tds_slopes(sesme, root_outdir):
    fig, ax = plt.subplots()

    tds = sesme.tscan.tds
    tds_slope = sesme.tscan.tds_slope

    ax.plot(tds, tds_slope * 1e-6)
    ax.set_xlabel("TDS Magic Number / %")
    ax.set_ylabel(r"TDS Calibration Slope / $\mathrm{\mu{}mps^{-1}}$")

    if root_outdir is not None:
        fig.savefig(root_outdir / "tds-calibration-slopes.png")


def plot_tds_voltages(sesme, root_outdir):
    fig, ax = plt.subplots()

    tds = sesme.tscan.tds
    tds_voltage = abs(sesme.tscan.tds_voltage * 1e-6)

    ax.plot(tds, tds_voltage)
    ax.set_xlabel("TDS Magic Number / %")
    ax.set_ylabel(r"$|V_\mathrm{TDS}|$ / MV")

    if root_outdir is not None:
        fig.savefig(root_outdir / "derived-tds-voltages.png")
