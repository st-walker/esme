"""Console script for esme."""

import logging
from pathlib import Path

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
    add_metadata_to_pcls_in_toml,
    tscan_config_from_scan_config_file,
    dscan_config_from_scan_config_file
)

from esme.plot import (
    dump_full_scan,
    plot_measured_central_widths,
    plot_quad_strengths,
    plot_scans,
    plot_tds_calibration,
    pretty_parameter_table,
    compare_results,
    plot_tds_set_point_vs_readback,
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
            kwargs['help'] = help + (
                ' NOTE: This argument is mutually exclusive with '
                f' arguments: [{ex_str}].'
            )
        super(MutuallyExclusiveOption, self).__init__(*args, **kwargs)

    def handle_parse_result(self, ctx, opts, args):
        if self.mutually_exclusive.intersection(opts) and self.name in opts:
            raise UsageError(
                "Illegal usage: `{}` is mutually exclusive with "
                "arguments `{}`.".format(
                    self.name,
                    ', '.join(self.mutually_exclusive)
                )
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(
            ctx,
            opts,
            args
        )



@group()
@option("--debug", is_flag=True, help="Run all subcommands in debug mode")
@option("--single-threaded", is_flag=True, help="Run in a single process")
def main(debug, single_threaded):
    """Main entrypoint."""
    echo("esme-xfel")
    echo("=" * len("esme-xfel"))
    echo(
        "Automatic calibration, data taking and analysis for" " uncorrelated energy spread measurements at the EuXFEL"
    )

    if single_threaded:
        esme.analysis.MULTIPROCESSING = False

    if debug:
        logging.getLogger("esme.analysis").setLevel(logging.DEBUG)
        logging.getLogger("esme.plot").setLevel(logging.DEBUG)
        logging.getLogger("esme.lattice").setLevel(logging.DEBUG)
        logging.getLogger("esme.calibration").setLevel(logging.DEBUG)
        logging.getLogger("esme.inout").setLevel(logging.DEBUG)
        logging.getLogger("esme.measurement").setLevel(logging.DEBUG)


@main.command(no_args_is_help=True)
@argument("scan-tomls", nargs=-1)
@option(
    "--simple", "-s", is_flag=True, help="Calculate the energy spread without accounting for the impact of the TDS."
)
@option("--latex", is_flag=True)
def calc(scan_tomls, simple, latex):
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
            print(pretty_parameter_table(slice_energy_spread_measurements[0], latex))
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
            plot_tds_calibration(sesme, root_outdir)
            with (root_outdir / "parameters.txt").open("w") as f:
                f.write(pretty_parameter_table(sesme))
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


@main.command(no_args_is_help=True, hidden=True)
@argument("ftoml", nargs=1)
@option("--new-columns", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["to_snapshot"])
@option("--to-snapshots", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["new_columns"])
@option("--alle", is_flag=True, cls=MutuallyExclusiveOption, mutually_exclusive=["new_columns", "to_snapshot"])
def fix(ftoml, new_columns, to_snapshots, alle):
    """Update metadata in old pcl snapshot files to match newer formats"""
    if new_columns:
        echo(f"Adding columns to files in {ftoml}.")
        add_metadata_to_pcls_in_toml(ftoml)
    elif to_snapshots:
        echo(f"Converting pcl files in {ftoml} to SetpointSnapshot instances.")
        toml_dfs_to_setpoint_snapshots(ftoml)
    elif alle:
        add_metadata_to_pcls_in_toml(ftoml)
        toml_dfs_to_setpoint_snapshots(ftoml)
    else:
        raise UsageError("No flags provided")



@main.command(no_args_is_help=True, hidden=True)
@argument("ftoml", nargs=1)
def compsp(ftoml):
    """Compare TDS amplitude readbacks to setpoints"""
    dscan, tscan, bscan = scan_files_from_toml(ftoml)
    title = title_from_toml(ftoml)
    plot_tds_set_point_vs_readback(dscan, tscan, title=title)


@main.command(no_args_is_help=True)
@argument("name", nargs=1)
@option("--dispersion", is_flag=True, help="Just measure the dispersion (used for debugging purposes)")
@option("--bscan", is_flag=True, help="Only do the beta scan")
@option("--dscan", is_flag=True, help="Only do the dispersion scan")
@option("--tscan", is_flag=True, help="Only do the TDS scan")
@option("--config", help="Explicitly set the config to be used.", type=Path)
# @argument("--continue", n) # continue_ file...
def measure(name, dispersion, bscan, dscan, tscan, config):
    """Measure the slice energy spread in the EuXFEL on the command line"""

    config = find_scan_config(config, "./scan.toml")

    if dispersion:
        dispersion_measurer = make_dispersion_measurer(config)
        dispersion, dispersion_unc = dispersion_measurer.measure()
        print(f"{dispersion=}±{dispersion_unc}")
        return

    measurer = make_measurement_runner(name, config)

    bg_shots = config["nbackground"]
    beam_shots = config["nbeam"]

    if dscan:
        measurer.dispersion_scan(bg_shots=bg_shots, beam_shots=beam_shots)
    if tscan:
        measurer.tds_scan(bg_shots=bg_shots, beam_shots=beam_shots)
    # if bscan:
    #     measurer.beta_scan(bg_shots=bg_shots, beam_shots=beam_shots)

    if not (dscan or tscan or bscan):
        measurer.run(bg_shots=bg_shots, beam_shots=beam_shots)


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
    """Delete a .pcl snapshots file and all images it refers to"""
    if not pcl_files:
        raise UsageError("No .pcl files provided")
    for fpcl in pcl_files:
        if imname is None:
            rm_pcl(fpcl, dry_run)
        else:
            rm_ims_from_pcl(fpcl, imname, dry_run)


@main.command(no_args_is_help=True)
def gui():
    """Start the measurement GUI"""
    pass


@main.command(no_args_is_help=True)
@argument("fscan", nargs=1)
@option("--i1d", is_flag=True)
@option("--optics", is_flag=True)
def sim(fscan, i1d, optics):
    from .sim import run_i1d_dispersion_scan

    # tscan_conf = tscan_config_from_scan_config_file(fscan)
    dscan_conf = dscan_config_from_scan_config_file(fscan)

    run_i1d_dispersion_scan(dscan_conf)


if __name__ == "__main__":
    main()  # pragma: no cover, pylint: disable=no-value-for-parameter
