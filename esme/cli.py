"""Console script for esme."""

from pathlib import Path

import click
import matplotlib.pyplot as plt

from esme.analysis import calculate_energy_spread_simple
from esme.inout import load_config
from esme.plot import (dump_full_scan, plot_measured_central_widths, plot_scans,
                       pretty_beam_parameter_table, pretty_beam_parameter_table,
                       pretty_measured_beam_sizes)


def preamble():
    click.echo("esme-xfel")
    click.echo("=" * len("esme-xfel"))
    click.echo(
        "Automatic calibration, data taking and analysis for" " uncorrelated energy spread measurements at the EuXFEL"
    )


# @click.argument("filelist", nargs=1)
# @click.command()
@click.group()
def main():
    """Main entrypoint."""
    preamble()


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
@click.option("--dump_images", "-d", is_flag=True, help="Dump all images used in the calculation to file")
@click.option("--tables", "-t", is_flag=True)
def calc(scan_inis, simple, dump_images, tables):

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

        # if dump_images:
        #     for fname, sesme in zip(scan_inis, slice_energy_spread_measurements):
        #     for ini_file in scan_ini:
        #         root_outdir = Path(ini_file).resolve().parent / (Path(ini_file).stem + "-images")
        #         dump_full_scan(dispersion_scan, tds_scan, root_outdir)

        # plot_scans(dispersion_scan, tds_scan)
    # if tables:
    #     from IPython import embed; embed()

    plt.show()


# class CacheRunner

# @main.command()
# @click.argument("scan-ini", nargs=-1)
# def plot(scan_ini):
#     for ini_file in scan_ini:
#         dscan, tscan = load_ini(ini_file)

#         plot_measured_central_widths(dscan, tscan)
#         plt.show()

if __name__ == "__main__":
    main()  # pragma: no cover
