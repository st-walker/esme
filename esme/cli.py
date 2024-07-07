"""Console script for esme."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from click import Option
from click import Path as CPath
from click import UsageError, argument, echo, group, option
import sys

import esme.analysis as ana
import esme.plot as plot
from esme.analysis import SetpointDataFrame
from esme.gui.explorer import start_explorer_gui
from esme.gui.tds_calibrator import start_calibration_explorer_gui
from esme.load import load_result_directory
from esme.plot import pretty_parameter_table
import esme.gui.widgets.common as wcommon

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


class MutuallyExclusiveOption(Option):
    # From https://stackoverflow.com/a/37491504
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop("mutually_exclusive", []))
        help = kwargs.get("help", "")
        if self.mutually_exclusive:
            ex_str = ", ".join(self.mutually_exclusive)
            kwargs["help"] = help + (
                " NOTE: This argument is mutually exclusive with "
                f" arguments: [{ex_str}]."
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(self.name, ", ".join(self.mutually_exclusive))
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(ctx, opts, args)


def calculate_bunch_length(setpoint: SetpointDataFrame):
    bunch_length = ana.true_bunch_length_from_df(setpoint)
    return bunch_length


@group()
@option("--debug", is_flag=True, help="Run all subcommands in debug mode.")
@option("--vxfel", is_flag=True, help="Run all commands in virtual XFEL address space.")
@option("--profile", is_flag=True)
def main(debug, profile, vxfel):
    """Main entrypoint."""
    echo("esme-xfel")
    echo("=" * len("esme-xfel"))
    echo(
        "Automatic calibration, data taking and analysis for"
        " uncorrelated energy spread measurements at the EuXFEL"
    )

    if debug:
        logging.getLogger("esme.analysis").setLevel(logging.DEBUG)
        logging.getLogger("esme.plot").setLevel(logging.DEBUG)
        logging.getLogger("esme.lattice").setLevel(logging.DEBUG)
        logging.getLogger("esme.calibration").setLevel(logging.DEBUG)
        logging.getLogger("esme.inout").setLevel(logging.DEBUG)
        logging.getLogger("esme.measurement").setLevel(logging.DEBUG)
        logging.getLogger("esme.sim").setLevel(logging.DEBUG)
        logging.getLogger("esme.simplot").setLevel(logging.DEBUG)
        logging.getLogger("esme.explorer").setLevel(logging.DEBUG)

        logging.getLogger("ocelot.utils.fel_track").setLevel(logging.DEBUG)
    
    wcommon.USE_VIRTUAL_XFEL_ADDRESSES = vxfel

    if profile:
        import atexit
        import cProfile
        import io
        import pstats

        print("Profiling...")
        pr = cProfile.Profile()
        pr.enable()

        def exit():
            pr.disable()
            print("Profiling completed")
            s = io.StringIO()
            pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats()
            print(s.getvalue())

        atexit.register(exit)


@main.command()
def calib():
    from esme.gui.tds_calibrator import start_bolko_tool

    start_bolko_tool()


@main.command(no_args_is_help=True)
@argument("dirname", nargs=1, type=CPath(exists=True, file_okay=False, path_type=Path))
@option("--analysis", is_flag=True)
@option("--widths", is_flag=True)
@option("--window", type=int, help="window FULL SIZE", default=21)
@option("--parabolic", is_flag=True)
def debug(dirname, analysis, widths, window, parabolic):
    measurement = load_result_directory(dirname)

    outdir = Path(f"ANALYSIS-{dirname.name}")

    if analysis:
        fit_type_string = "-parabolic-fit" if parabolic else "-gaussian-fit"
        image_outdir = outdir / f"image-analysis-{window=}{fit_type_string}"
        voltage = measurement.tscan.voltages() * 1e-6
        dispersion = measurement.dscan.dispersions()
        beta = measurement.bscan.betas()

        plot.break_scan_down(
            measurement.tscan,
            image_outdir,
            voltage,
            plot.VOLTAGE_LABEL,
            "tscan",
            window=window,
            parabolic=parabolic,
        )
        plot.break_scan_down(
            measurement.dscan,
            image_outdir,
            dispersion,
            plot.ETA_LABEL,
            "dscan",
            window=window,
        )
        plot.break_scan_down(
            measurement.bscan,
            image_outdir,
            beta,
            plot.BETA_LABEL,
            "bscan",
            window=window,
        )

    # plot.plot_tds_voltages(measurement, outdir)
    # plot.plot_amplitude_setpoints_with_readbacks(measurement, outdir)
    # plot.plot_streaking_parameters(measurement, outdir)
    # plot.plot_r12_streaking(measurement, outdir)
    # plot.plot_streaking_plane_beamsizes(measurement, outdir)
    plot.plot_slice_length(measurement, outdir)
    # plot.plot_apparent_bunch_lengths(measurement, outdir)
    plot.plot_true_bunch_lengths(measurement, outdir)

    import matplotlib.pyplot as plt

    plt.show()


@main.command()
@option(
    "--calibration",
    required=False,
    nargs=1,
    type=CPath(exists=True, dir_okay=False, path_type=Path),
)
@argument(
    "dirname", required=False, type=CPath(exists=True, file_okay=False, path_type=Path)
)
def explorer(dirname, calibration):
    start_explorer_gui(dirname, calibration)


# @main.command()
# @argument("calib_file", required=True)
# # def calibration(calib_file):


@main.command()
@argument("calib_file", required=True)
def calibration(calib_file):
    start_calibration_explorer_gui(calib_file)


@main.command(no_args_is_help=True)
@argument("dirname", nargs=1, type=CPath(exists=True, file_okay=False, path_type=Path))
@option("--with-sigr", is_flag=True)
@option("--with-en", is_flag=True)
@option("--derr")
@option("--simple", is_flag=True)
def process(dirname, with_sigr, with_en, derr, simple):
    # Calculate bunch length for maximum streaking case.
    measurement = load_result_directory("./")  # list(dirname.glob("*m.pkl")),
    #  image_dir=dirname,
    #  image_address=image_address,
    #  energy_address=energy_address)
    calib = load_calibration_from_result_directory("./")
    avmapping = calib.mapping()
    fit_df, beam_df = ana.process_measurment_dfs(measurement, avmapping)

    print(pretty_parameter_table(fit_df, beam_df))


def _simple_calc(setpoint, sigma_r=28e-6):
    mean_width = ana.pixel_widths_from_setpoint(setpoint)

    width_m2, error_m2 = ana.transform_pixel_widths(
        [mean_width.n], [mean_width.s], pixel_units="m", to_variances=True
    )

    sigma_r2 = sigma_r**2
    from uncertainties import ufloat
    from uncertainties.umath import sqrt as usqrt

    unc_width_m2 = ufloat(width_m2, error_m2)

    dispersion = setpoint.dispersion
    energy = setpoint.energy * 1e6
    from ocelot.common.globals import m_e_GeV

    emit = 0.43e-6
    beta = 0.6
    egamma = energy / (m_e_GeV * 1e9)

    sigma_b2 = beta * emit / egamma

    sigma_e = usqrt(unc_width_m2 - sigma_r2 - sigma_b2) * energy / dispersion

    print(sigma_e)
    return sigma_e


@main.command()
@argument("dirname")
def show(dirname):
    measurement, ofp = load_result_directory(dirname)
    for setpoint in measurement.tscan:
        for path in setpoint.image_full_paths:
            np.load


# @main.command()
# @argument("pcl", nargs=1)
# def display(pcl):
#     from esme.gui.scannerpanel import display
#     display(pcl)

# from IPython import embed; embed()


@main.command(no_args_is_help=True)
@argument("pcl-files", nargs=-1)
@option(
    "--dry-run",
    "-n",
    is_flag=True,
    help="Don’t actually remove any file(s). Instead, just show if they exist in the index and would otherwise be removed by the command.",
)
@option("--imname", "-i", multiple=True)
def rm(pcl_files, imname, dry_run):
    """Delete a .pcl snapshots file and all images it refers to, or
    delete a specific snapshot within a .pcl file."""
    if not pcl_files:
        raise UsageError("No .pcl files provided")
    for fpcl in pcl_files:
        if imname is None:
            rm_pcl(fpcl, dry_run)
        else:
            rm_ims_from_pcl(fpcl, imname, dry_run)



@main.command()
def gui():
    """Start the measurement GUI"""
    from esme.gui import start_lps_gui

    start_lps_gui()

@main.command()
def tds():
    from esme.gui.calibrator import start_tds_calibrator
    start_tds_calibrator(sys.argv)

@main.command()
@argument("dirname", nargs=1, type=CPath(exists=True, file_okay=False, path_type=Path))
def optics(dirname):
    # Assuming matching at MATCH.52.I1, we track and backtrack from
    # that point to get the full optics...
    import matplotlib.pyplot as plt

    from esme.optics import optics_from_measurement_df

    fig, ax = plt.subplots()
    for fpkl in dirname.glob("*.pkl"):
        df = pd.read_pickle(fpkl)
        twiss, mlat = optics_from_measurement_df(df)
        outname = f"{fpkl.stem}-optics.pkl"
        outdir = dirname / "optics"
        outdir.mkdir(exist_ok=True)
        twiss.to_pickle(outdir / outname)

    #     ax.plot(twiss.s, twiss.beta_x)
    # plt.show()
    # from IPython import embed; embed()


@main.command()
@argument("seqname", nargs=1)
def taskomat(seqname):
    from esme.gui.widgets.sequence import TaskomatSequenceDisplay
    from esme.control.taskomat import Sequence
    from PyQt5.QtWidgets import QApplication
    from esme.gui.widgets.common import make_default_doocs_interface
    import sys
    app = QApplication(sys.argv)

    sequence = Sequence(seqname, di=make_default_doocs_interface())
    display = TaskomatSequenceDisplay(sequence)
    display.setWindowTitle("Taskomat Sequence Runner")
    display.show()
    display.raise_()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()  # pragma: no cover, pylint: disable=no-value-for-parameter
