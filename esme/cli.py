"""Console script for esme."""

import click
import matplotlib.pyplot as plt

from esme.files import load_ini
from esme.plot import plot_dispersion_scan, plot_tds_scan


# @click.argument("filelist", nargs=1)
@click.command()
@click.argument('scan-ini')
def main(scan_ini):
    """Main entrypoint."""
    click.echo("esme-xfel")
    click.echo("=" * len("esme-xfel"))
    click.echo(
        "Automatic calibration, data taking and analysis for uncorrelated energy spread measurements at the EuXFEL"
    )

    dispersion_scan, tds_scan = load_ini(scan_ini)

    plot_dispersion_scan(dispersion_scan)
    plot_tds_scan(tds_scan)
    plt.show()


if __name__ == "__main__":
    main()  # pragma: no cover
