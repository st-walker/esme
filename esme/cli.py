"""Console script for esme."""

import logging
from pathlib import Path
from contextlib import ExitStack

from click import command, option, Option, UsageError, group, argument, echo, UsageError

from esme.analysis import calculate_energy_spread_simple
import esme.analysis
from esme.inout import (
    load_config,
    scan_files_from_toml,
    title_from_toml,
    make_measurement_runner,
    make_dispersion_measurer,
    find_scan_config,
    rm_pcl,
    rm_ims_from_pcl,
    toml_dfs_to_setpoint_snapshots,
    i1_dscan_config_from_scan_config_file,
    b2_dscan_config_from_scan_config_file,
    i1_tds_voltages_from_scan_config_file,
    get_config_sample_sizes,
)

from . import inout

from esme.plot import (
    dump_full_scan,
    plot_measured_central_widths,
    plot_quad_strengths,
    plot_scans,
    plot_tds_calibration,
    write_pixel_widths,
    pretty_parameter_table,
    compare_results,
    plot_tds_set_point_vs_readback,
    formatted_parameter_dfs,
)


logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


class MutuallyExclusiveOption(Option):
    # From https://stackoverflow.com/a/37491504
    def __init__(self, *args, **kwargs):
        self.mutually_exclusive = set(kwargs.pop('mutually_exclusive', []))
        help = kwargs.get('help', '')
        if self.mutually_exclusive:
            ex_str = ', '.join(self.mutually_exclusive)
            kwargs['help'] = help + (' NOTE: This argument is mutually exclusive with ' f' arguments: [{ex_str}].')
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(self.name, ', '.join(self.mutually_exclusive))
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(ctx, opts, args)


@group()
@option("--debug", is_flag=True, help="Run all subcommands in debug mode")
@option("--single-threaded", is_flag=True, help="Run in a single process")
def main(debug, single_threaded):
    """Main entrypoint."""
    echo("esme-xfel")
    echo("=" * len("esme-xfel"))
    echo("Automatic calibration, data taking and analysis for" " uncorrelated energy spread measurements at the EuXFEL")

    if single_threaded:
        esme.analysis.MULTIPROCESSING = False

    if debug:
        logging.getLogger("esme.analysis").setLevel(logging.DEBUG)
        logging.getLogger("esme.plot").setLevel(logging.DEBUG)
        logging.getLogger("esme.lattice").setLevel(logging.DEBUG)
        logging.getLogger("esme.calibration").setLevel(logging.DEBUG)
        logging.getLogger("esme.inout").setLevel(logging.DEBUG)
        logging.getLogger("esme.measurement").setLevel(logging.DEBUG)
        logging.getLogger("esme.sim").setLevel(logging.DEBUG)
        logging.getLogger("esme.simplot").setLevel(logging.DEBUG)
        logging.getLogger("ocelot.utils.fel_track").setLevel(logging.DEBUG)


@main.command(no_args_is_help=True)
@argument("scan-tomls", nargs=-1)
@option(
    "--simple", "-s", is_flag=True, help="Calculate the energy spread without accounting for the impact of the TDS."
)
@option("--latex", is_flag=True)
def calculate(scan_tomls, simple, latex):
    """Calculate the slice energy spread using a toml analysis file"""

    slice_energy_spread_measurements = [load_config(fname) for fname in scan_tomls]

    # sesme = load_config() # Load the slice energy spread measurement instance.

    echo("Calculation of energy spread")
    if simple:
        for fname, sesme in zip(scan_tomls, slice_energy_spread_measurements):
            espread_ev, error_ev = calculate_energy_spread_simple(sesme.dscan)
            espread_kev = espread_ev * 1e-3
            error_kev = error_ev * 1e-3
            print(fname)
            print(f"({espread_kev}±{error_kev})keV")
    else:
        if len(slice_energy_spread_measurements) == 1:
            fit_df, beam_df = formatted_parameter_dfs(slice_energy_spread_measurements[0], latex=latex)
            if latex is False:
                ftoml = Path(scan_tomls[0])
                name = ftoml.stem
                beam_file_name = Path(f"{name}-beam.csv")
                fit_file_name = Path(f"{name}-fit.csv")

                with beam_file_name.open("w") as f:
                    f.write(beam_df.to_csv())

                with fit_file_name.open("w") as f:
                    f.write(fit_df.to_csv())

            else:
                print(pretty_parameter_table(fit_df, beam_df, latex))
        else:
            print(compare_results(slice_energy_spread_measurements, latex))


@main.command(no_args_is_help=True)
@argument("scan-tomls", nargs=-1)
@option("--dump-images", "-d", is_flag=True, help="Dump all images used in the calculation to file")
@option("--widths", "-w", is_flag=True, help="Dump all images used in the calculation to file")
@option("--magnets", "-m", is_flag=True)
@option("--calibration", "-c", is_flag=True)
@option("--alle", is_flag=True)
@option("--save", "-s", is_flag=True)
def plot(scan_tomls, dump_images, widths, magnets, alle, calibration, save):
    """Make lots of different plots from toml analysis files"""
    slice_energy_spread_measurements = [load_config(fname) for fname in scan_tomls]
    for fname, sesme in zip(scan_tomls, slice_energy_spread_measurements):
        root_outdir = None
        root_outdir = Path(fname).resolve().parent / (Path(fname).stem + "-images")
        echo(f"Writing to outdir: {root_outdir}")
        if alle:
            root_outdir = Path(fname).resolve().parent / (Path(fname).stem + "-images")
            echo(f"Writing plots to {root_outdir}")
            dump_full_scan(sesme, root_outdir)
            plot_measured_central_widths(sesme, root_outdir, show=False)
            plot_scans(sesme, root_outdir)
            plot_quad_strengths(sesme, root_outdir)
            # plot_tds_calibration(sesme, root_outdir)
            write_pixel_widths(sesme, root_outdir)
            with ExitStack() as stack:
                f1 = stack.enter_context(open(root_outdir / "parameters.txt", "w"))
                f2 = stack.enter_context(open(root_outdir / "fit.csv", "w"))
                f3 = stack.enter_context(open(root_outdir / "beam.csv", "w"))

                params = sesme.all_fit_parameters()
                fit_df = params.fit_parameters_to_df()
                f2.write(fit_df.to_csv())
                beam_df = params.beam_parameters_to_df()
                f3.write(beam_df.to_csv())
                f1.write(pretty_parameter_table(fit_df, beam_df))

        elif calibration:
            plot_tds_calibration(sesme, root_outdir)
        elif dump_images:
            dump_full_scan(sesme, root_outdir)
        elif widths:
            plot_measured_central_widths(sesme, root_outdir)
        elif magnets:
            plot_quad_strengths(sesme, root_outdir)
        else:
            plot_scans(sesme, root_outdir)


@main.command(no_args_is_help=True)
@option("--i1", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["b2"])
@option("--b2", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["i1"])
@argument("fname", nargs=1)
def snapshot(model, i1, b2):
    """Take a snapshot of the file up to the given point in the
    machine (either I1 or B2), write it to file, and then exit"""
    from esme.measurement import I1EnergySpreadMeasuringMachine, B2EnergySpreadMeasuringMachine

    if i1:
        machine = I1EnergySpreadMeasuringMachine(fname)
    elif b2:
        machine = B2EnergySpreadMeasuringMachine(fname)
    else:
        raise UsageError("Unrecognised model string")

    df = machine.get_machine_snapshot()
    df.to_csv(fname)


# @main.command(no_args_is_help=True, hidden=True)
# @argument("ftoml", nargs=1)
# @argument("dispersion", nargs=1)
# @option("--i1", is_flag=True)
# @option("--b2", is_flag=True)
# def setpoint(ftoml, dispersion, i1, b2):
#     if i1:
#         dconf = i1_dscan_config_from_scan_config_file(ftoml)
#         print("I1 Dispersion Scan Setpoints")
#         print("Reference Setting:")


@main.command(no_args_is_help=True)
@option("--b2", is_flag=True)
@option("--i1", is_flag=True)
@option("--dispersion", is_flag=True, help="Just measure the dispersion (used for debugging purposes)")
@option("--bscan", is_flag=True, help="Only do the beta scan")
@option("--dscan", is_flag=True, help="Only do the dispersion scan")
@option("--tscan", is_flag=True, help="Only do the TDS scan")
@option("--config", help="Explicitly set the config to be used.", type=Path)
@argument("outdir", nargs=1, type=Path)
# @argument("--continue", n) # continue_ file...
def measure(dispersion, bscan, dscan, tscan, config, b2, i1, outdir):
    """Measure the slice energy spread in the EuXFEL on the command line"""
    config = find_scan_config(config, "./scan.toml")

    if dispersion:
        dispersion_measurer = make_dispersion_measurer(config)
        dispersion, dispersion_unc = dispersion_measurer.measure()
        print(f"{dispersion=}±{dispersion_unc}")
        return

    if b2:
        model = "b2"
    elif i1:
        model = "i1"

    measurer = make_measurement_runner(config, model, outdir=outdir)

    bg_shots, beam_shots = get_config_sample_sizes(config)

    measure_dispersion = False
    if dscan:
        measurer.dispersion_scan(bg_shots=bg_shots, beam_shots=beam_shots, measure_dispersion=measure_dispersion)
    if tscan:
        measurer.tds_scan(bg_shots=bg_shots, beam_shots=beam_shots, measure_dispersion=measure_dispersion)
    if bscan:
        measurer.beta_scan(bg_shots=bg_shots, beam_shots=beam_shots, measure_dispersion=measure_dispersion)

    basepath = "./"
    import toml

    if not (dscan or tscan or bscan):
        tscan_files, dscan_files = measurer.run(bg_shots=bg_shots, beam_shots=beam_shots)
        # from IPython import embed; embed()
        template = {
            'title': 'Jan 2023 Energy Spread Measurement (first of three)',
            'optics': {
                'tds': {
                    'bety': 4.3,
                    'alfy': 1.9,
                    'wavenumber': 62.88,
                    'length': 0.7,
                    'calibration': {
                        'percentages': [11, 15, 19, 23],
                        'tds_slopes': [256, 343, 455, 593],
                        'tds_slope_units': 'um/ps',
                        'screen_name': 'OTRC.64.I1D',
                        'dispersion_setpoint': 1.2,
                    },
                },
                'screen': {'betx': 0.6},
            },
            'data': {
                'basepath': str(basepath),
                'screen_channel': 'XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ',
                'bad_images': [],
                'dscan': {'fnames': [str(x) for x in dscan_files]},
                'tscan': {'fnames': [str(x) for x in tscan_files]},
            },
        }
        with open(str(outdir) + ".toml", "w") as f:
            toml.dump(template, f)


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
@argument("ftoml", nargs=1)
@option("--replay", "-r", nargs=1, help="scan.toml file to replay output from")
def gui(ftoml, replay):
    """Start the measurement GUI"""
    from esme.gui import start_gui

    start_gui(ftoml, debug_mode=True, replay=replay)


@main.command()
def tds():
    from esme.gui import start_tds_gui

    start_tds_gui()


@main.command()
def error(
    i1,
    b2,
    espread,
    simulations,
):
    pass


@main.command(no_args_is_help=True)
@argument("fscan", nargs=1)
@argument("parray", nargs=1, required=False)
@option("--outdir", "-o", nargs=1, default="./", show_default=True, help="where to write output files to")
@option("--i1", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["b2"])
@option("--b2", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["i1"])
@option("--dscan", is_flag=True)
@option("--tscan", is_flag=True)
@option("--escan", is_flag=True)
@option("--fast", is_flag=True)
@option("--optics", is_flag=True)
@option("--physics", is_flag=True)
def sim(fscan, i1, b2, dscan, tscan, escan, parray, outdir, fast, optics, physics):
    """This is basically just for checking optics and showing the basic functions work."""
    # from .sim import run_i1_dispersion_scan
    from . import simplot
    from . import sim

    if i1:
        i1_dscan_conf = i1_dscan_config_from_scan_config_file(fscan)
        i1_tscan_voltages = i1_tds_voltages_from_scan_config_file(fscan)

    if i1 and optics and not parray:
        simplot.a1_to_i1_design_optics()
        simplot.qi52_to_i1_dscan_optics(i1_dscan_conf)
        simplot.a1_to_i1_piecewise_measurement_optics(i1_dscan_conf)

    elif i1 and optics and parray:
        simplot.check_a1_to_i1_design_optics_tracking(parray, outdir)
        # simplot.check_a1_q52_measurement_optics_tracking(parray, outdir)
        # simplot.dscan_piecewise_tracking_optics(parray, i1_dscan_conf, outdir)
        # simplot.dscan_piecewise_tracking_optics(parray, i1_dscan_conf, outdir,
        #                                         do_physics=physics)
    elif i1 and parray:
        i1sim = sim.I1SimulatedEnergySpreadMeasurement(parray, i1_dscan_conf, i1_tscan_voltages)
        i1sim.write_scans(outdir)
        # sim.run_i1_dispersion_scan(i1_dscan_conf, parray, outdir)

    if b2:
        b2_dscan_conf = inout.b2_dscan_config_from_scan_config_file(fscan)
        b2_tscan_voltages = inout.b2_tds_voltages_from_scan_config_file(fscan)

        if optics:
            # b2sim = sim.B2SimulatedEnergySpreadMeasurement(parray,
            #                                                  b2_dscan_conf,
            #                                                  b2_tscan_voltages)

            simplot.plot_b2_design_optics(b2_dscan_conf, b2_tscan_voltages)

            simplot.bolko_optics_comparison(b2_dscan_conf, b2_tscan_voltages)

            simplot.gun_to_b2_bolko_optics(b2_dscan_conf, b2_tscan_voltages)

            simplot.gun_to_b2_dispersion_scan_design_energy(b2_dscan_conf, b2_tscan_voltages)
            simplot.gun_to_b2_piecewise_dispersion_scan_optics(b2_dscan_conf, b2_tscan_voltages)
            simplot.gun_to_b2_tracking_piecewise_optics(b2_dscan_conf, b2_tscan_voltages, parray)
            simplot.gun_to_b2_tracking_central_slice_optics(
                b2_dscan_conf, b2_tscan_voltages, parray, outdir=outdir, do_physics=physics
            )

        else:
            simplot.gun_to_b2_tracking_central_slice_optics(
                b2_dscan_conf, b2_tscan_voltages, parray, outdir=outdir, do_physics=True
            )


# def make_anaconf(outfiles):
#     pass

if __name__ == "__main__":
    main()  # pragma: no cover, pylint: disable=no-value-for-parameter
