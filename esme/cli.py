"""Console script for esme."""

import sys
import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt

from esme.analysis import ScanMeasurement, calculate_energy_spread_simple
from esme.inout import load_config, add_metadata_to_pcls_in_toml
from esme.plot import (
    dump_full_scan,
    plot_measured_central_widths,
    plot_quad_strengths,
    plot_scans,
    plot_tds_calibration,
    pretty_beam_parameter_table,
    show_before_after_processing,
)

logging.basicConfig()


@click.group()
@click.option("--debug", is_flag=True)
def main(debug):
    """Main entrypoint."""
    click.echo("esme-xfel")
    click.echo("=" * len("esme-xfel"))
    click.echo(
        "Automatic calibration, data taking and analysis for" " uncorrelated energy spread measurements at the EuXFEL"
    )

    if debug:
        logging.getLogger("esme.analysis").setLevel(logging.DEBUG)
        logging.getLogger("esme.plot").setLevel(logging.DEBUG)
        logging.getLogger("esme.lattice").setLevel(logging.DEBUG)
        logging.getLogger("esme.calibration").setLevel(logging.DEBUG)
        logging.getLogger("esme.inout").setLevel(logging.DEBUG)


@main.command()
@click.argument('scan-ini')
def optics(scan_ini):
    click.echo("esme-xfel")
    click.echo("Loading optics")


@main.command()
@click.argument("scan-inis", nargs=-1)
@click.option(
    "--simple", "-s", is_flag=True, help="Calculate the energy spread without accounting for the impact of the TDS."
)
def calc(scan_inis, simple):

    slice_energy_spread_measurements = [load_config(fname) for fname in scan_inis]

    # sesme = load_config() # Load the slice energy spread measurement instance.

    click.echo("Calculation of energy spread")
    if simple:
        for fname, sesme in zip(scan_inis, slice_energy_spread_measurements):
            espread_ev, error_ev = calculate_energy_spread_simple(sesme.dscan)
            espread_kev = espread_ev * 1e-3
            error_kev = error_ev * 1e-3
            print(fname)
            print(f"({espread_kev}±{error_kev})keV")
    else:
        for fname, sesme in zip(scan_inis, slice_energy_spread_measurements):
            # plot_scans(sesme)
            # plt.show()
            print(fname)
            print(pretty_beam_parameter_table(sesme))
            plt.show()


@main.command()
@click.argument("scan-inis", nargs=-1)
@click.option("--dump-images", "-d", is_flag=True, help="Dump all images used in the calculation to file")
@click.option("--widths", "-w", is_flag=True, help="Dump all images used in the calculation to file")
@click.option("--magnets", "-m", is_flag=True)
@click.option("--calibration", "-c", is_flag=True)
@click.option("--alle", is_flag=True)
@click.option("--save", "-s", is_flag=True)
def plot(scan_inis, dump_images, widths, magnets, alle, calibration, save):
    slice_energy_spread_measurements = [load_config(fname) for fname in scan_inis]
    for fname, sesme in zip(scan_inis, slice_energy_spread_measurements):
        root_outdir = None
        if alle:
            root_outdir = Path(fname).resolve().parent / (Path(fname).stem + "-images")
            click.echo(f"Writing plots to {root_outdir}")
            dump_full_scan(sesme, root_outdir)
            plot_measured_central_widths(sesme, root_outdir, show=False)
            plot_scans(sesme, root_outdir)
            plot_quad_strengths(sesme, root_outdir)
            plot_tds_calibration(sesme, root_outdir)
            with (root_outdir / "parameters.txt").open("w") as f:
                f.write(pretty_beam_parameter_table(sesme))
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


@main.command()
@click.argument("tcls", nargs=-1)
def diag(tcls):
    for f in tcls:
        sm = ScanMeasurement(f)
        width, error = sm.mean_central_slice_width_with_error()
        print(f"{f}: central width: width±error")
        for i in range(sm.nimages):
            show_before_after_processing(sm, i)
            plt.show()

# @main.command()
# @click.argument("--reset_quads", is_flag=True)
# @click.argument("--dx", nargs=1)
# @click.argument("--go", )
# def measurement(reset_quads, dx):
#     import esme.measurement as mea

#     if reset_quads:
#         mea.set_initial_optics()
#         sys.exit(0)

#     if dx:
#         mea.set_dscan_optics(dx)
#         sys.exit(0)

    # if take

@click.argument("ftoml", nargs=1)
def fix(ftoml):
    add_metadata_to_pcls_in_toml(ftoml)


if __name__ == "__main__":
    main()  # pragma: no cover
