"""Console script for esme."""

import glob
import logging
from contextlib import ExitStack
from pathlib import Path
import numpy as np
from scipy.constants import c

import pandas as pd
from click import Option
from click import Path as CPath
from click import UsageError, argument, echo, group, option
import yaml

import esme.analysis as ana
from esme.analysis import calculate_energy_spread_simple, MeasurementDataFrames, SetpointDataFrame
from esme.gui.hires import start_hires_gui
from esme.gui.common import DEFAULT_CONFIG_PATH
from esme.control.configs import get_scan_config_for_area
from esme.plot import pretty_parameter_table, formatted_parameter_dfs
from esme.optics import calculate_i1d_r34_from_tds_centre


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
                "arguments `{}`.".format(self.name, ', '.join(self.mutually_exclusive))
            )

        return super(MutuallyExclusiveOption, self).handle_parse_result(ctx, opts, args)



def calculate_bunch_length(setpoint: SetpointDataFrame):
    bunch_length = ana.true_bunch_length_from_df(setpoint)
    return bunch_length



@group()
@option("--debug", is_flag=True, help="Run all subcommands in debug mode")
def main(debug):
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
        logging.getLogger("ocelot.utils.fel_track").setLevel(logging.DEBUG)


@main.command(no_args_is_help=True)
@argument("dirname", nargs=1, type=CPath(exists=True, file_okay=False, path_type=Path))
@option("--with-sigr", is_flag=True)
@option("--with-en", is_flag=True)
@option("--derr")
@option("--simple", is_flag=True)
def process(dirname, with_sigr, with_en, derr, simple):

    scan_setpoint_path = dirname / "scan.yaml"
    with scan_setpoint_path.open("r") as f:
        scan_conf = yaml.safe_load(f)
    image_address = scan_conf["channels"]["image"]
    energy_address = scan_conf["channels"]["energy_at_screen"]
    
    ofp = ana.OpticsFixedPoints(beta_screen=scan_conf["beta_screen"],
                                beta_tds=scan_conf["beta_tds"],
                                alpha_tds=scan_conf["alpha_tds"])

    # Calculate bunch length for maximum streaking case.
    measurement = MeasurementDataFrames.from_filenames(list(dirname.glob("*m.pkl")),
                                                       image_dir=dirname,
                                                       image_address=image_address,
                                                       energy_address=energy_address)

    
    
    tscan_widths = {}
    for tscan_setpoint in measurement.tscans:
        tscan_widths[tscan_setpoint.voltage] = ana.pixel_widths_from_setpoint(tscan_setpoint)

    dscan_widths = {}
    for dscan_setpoint in measurement.dscans:
        dscan_widths[dscan_setpoint.dispersion] = ana.pixel_widths_from_setpoint(dscan_setpoint)

    bscan_widths = {}
    for bscan_setpoint in measurement.bscans:
        bscan_widths[dscan_setpoint.beta] = ana.pixel_widths_from_setpoint(bscan_setpoint)


    # in metres
    bunch_length = calculate_bunch_length(measurement.max_voltage_df())

    fitter = ana.SliceWidthsFitter(dscan_widths, tscan_widths)
    params = fitter.all_fit_parameters(measurement.energy() * 1e6, # to eV
                                       dscan_voltage=measurement.dscan_voltage(),
                                       tscan_dispersion=measurement.tscan_dispersion(),
                                       optics_fixed_points=ofp,
                                       sigma_z=(bunch_length.n, bunch_length.s))

    beam_df = params.beam_parameters_to_df()
    fit_df = params.fit_parameters_to_df()

    # If failed to reconstruct values...
    if np.isnan(beam_df.loc["sigma_i"]["values"]):
        sigma_e = _simple_calc(measurement.max_dispersion_sp())
        beam_df.loc["sigma_e"] = {"values": sigma_e.n,
                                  "errors": sigma_e.s,
                                  "alt_values": np.nan,
                                  "alt_errors": np.nan}
        
    beam_df.to_pickle(f"{dirname}-beam.pkl")
    fit_df.to_pickle(f"{dirname}-fit.pkl")

    # from IPython import embed; embed()

    fit_df, beam_df = formatted_parameter_dfs(params)

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
@argument("pcl", nargs=1)
def display(pcl):

    from esme.gui.scannerpanel import display

    display(pcl)
    
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
@argument("ftoml", nargs=1)
@option("--replay", "-r", nargs=1, help="scan.toml file to replay output from")
def gui(ftoml, replay):
    """Start the measurement GUI"""
    from esme.gui import start_gui

    start_gui(ftoml, debug_mode=True, replay=replay)


@main.command()
@argument("dirname", nargs=1, type=CPath(exists=True, file_okay=False, path_type=Path))
def optics(dirname):
    # Assuming matching at MATCH.52.I1, we track and backtrack from
    # that point to get the full optics...
    from esme.optics import optics_from_measurement_df

    import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    main()  # pragma: no cover, pylint: disable=no-value-for-parameter
