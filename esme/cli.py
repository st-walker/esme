"""Console script for esme."""

import click
import matplotlib.pyplot as plt

from esme.files import load_ini
from esme.plot import plot_scans
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
def calc(scan_ini, simple):
    dispersion_scan, tds_scan = load_ini(scan_ini)
    if simple:
        for ini_file in scan_ini:
            dispersion_scan, _ = load_ini(ini_file)
            espread_kev, error_kev = calculate_energy_spread_simple(dispersion_scan)
            print(ini_file)
            print(f"({espread_kev}±{error_kev})keV")
    else:
        plot_scans(dispersion_scan, tds_scan)
    plt.show()

if __name__ == "__main__":
    main()  # pragma: no cover
