"""Console script for esme."""

from pathlib import Path

import click
import matplotlib.pyplot as plt

from esme.files import load_ini
from esme.plot import plot_scans, dump_full_scan, plot_measured_central_widths
from esme.analysis import calculate_energy_spread_simple


def preamble():
    click.echo("esme-xfel")
    click.echo("=" * len("esme-xfel"))
    click.echo("Automatic calibration, data taking and analysis for"
               " uncorrelated energy spread measurements at the EuXFEL")


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
@click.argument("scan-ini", nargs=-1)
@click.option("--simple", "-s", is_flag=True,
                help="Calculate the energy spread without accounting for the impact of the TDS.")
@click.option("--dump_images", "-d", is_flag=True,
              help="Dump all images used in the calculation to file"
              )
def calc(scan_ini, simple, dump_images):
    dispersion_scan, tds_scan = load_ini(scan_ini)
    if simple:
        for ini_file in scan_ini:
            dispersion_scan, _ = load_ini(ini_file)
            espread_kev, error_kev = calculate_energy_spread_simple(dispersion_scan)
            print(ini_file)
            print(f"({espread_kev}±{error_kev})keV")
    else:
        if dump_images:
            for ini_file in scan_ini:
                root_outdir = Path(ini_file).resolve().parent / (Path(ini_file).stem + "-images")
                dump_full_scan(dispersion_scan, tds_scan, root_outdir)

        plot_scans(dispersion_scan, tds_scan)
    plt.show()

@main.command()
@click.argument("scan-ini", nargs=-1)
def plot(scan_ini):
    for ini_file in scan_ini:
        dscan, tscan = load_ini(ini_file)

        plot_measured_central_widths(dscan, tscan)
        plt.show()

if __name__ == "__main__":
    main()  # pragma: no cover
