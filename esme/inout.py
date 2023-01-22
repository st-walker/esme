#!/usr/bin/env python3

import pickle
import logging
import re
import os
from pathlib import Path
import contextlib
from typing import Union

import toml
import pandas as pd
import numpy as np

from esme.analysis import (
    DispersionScan,
    OpticalConfig,
    SliceEnergySpreadMeasurement,
    TDSScan,
    BetaScan,
    ScanMeasurement,
)
from esme.calibration import TDSCalibrator, TrivialTDSCalibrator
from esme.injector_channels import TDS_AMPLITUDE_READBACK_ADDRESS, DUMP_SCREEN_ADDRESS
from esme.measurement import MeasurementRunner, DispersionScanConfiguration, TDSScanConfiguration, DispersionMeasurer, SetpointSnapshots, ScanType

LOG = logging.getLogger(__name__)


class MissingMetadataInFileNameError(RuntimeError):
    pass


class MalformedESMEConfigFile(RuntimeError):
    pass


def _optics_config_from_dict(config: dict) -> OpticalConfig:
    tds = config["optics"]["tds"]
    screen_betax = config["optics"]["screen"]["betx"]

    return OpticalConfig(
        tds_bety=tds["bety"],
        tds_alfy=tds["alfy"],
        ocr_betx=screen_betax,
    )


def _files_from_config(config, scan_name) -> list[Path]:
    try:
        basepath = Path(config["data"]["basepath"])
    except KeyError:
        basepath = Path(".")

    fnames = config["data"][scan_name]["fnames"]
    paths = [basepath / f for f in fnames]
    return paths


def _dispersion_from_filename(fname: os.PathLike) -> float:
    path = Path(fname)
    match = re.search(r"Dx_[0-9]+", path.stem)

    if not match:
        raise MissingMetadataInFileNameError(fname)
    dx = float(match.group(0).split("Dx_")[1])
    return dx / 1000  # convert to metres


def tds_magic_number_from_filename(fname: os.PathLike) -> int:
    path = Path(fname)
    match = re.search(r"tds_[0-9]+", path.stem)

    if not match:
        raise MissingMetadataInFileNameError(fname)
    tds_magic_number = int(match.group(0).split("tds_")[1])

    return tds_magic_number


def _beta_from_filename(fname: os.PathLike) -> float:
    path = Path(fname)
    match = re.search(r"beta_x_[0-9]+(.[0-9]*)?", path.stem)
    if not match:
        raise MissingMetadataInFileNameError(fname)
    beta = float(match.group(0).split("beta_x_")[1])
    return beta


def add_metadata_to_pickled_df(fname, force_dx=None):

    LOG.debug(f"Adding metadata to pickled file: {fname}")

    df = pd.read_pickle(fname)
    if isinstance(df, SetpointSnapshots):
        LOG.debug(f"{fname} is a pickled SetpointSnapshots instance, not a raw df.")
        return
    elif df is None:
        import ipdb; ipdb.set_trace()

    tds_amplitude = tds_magic_number_from_filename(fname)

    if not force_dx:
        dispersion = _dispersion_from_filename(fname)
    else:
        dispersion = force_dx

    try:
        beta = _beta_from_filename(fname)
    except MissingMetadataInFileNameError:
        beta = None

    LOG.debug(f"Adding dx, tds: {dispersion=}, {tds_amplitude=}")
    df["MY_SCREEN_DX"] = dispersion

    if beta:
        LOG.info(f"Adding BETA metadata to pickled file: {beta=}")
        df["MY_SCREEN_BETA"] = beta
    else:
        with contextlib.suppress(KeyError):
            del df["MY_SCREEN_BETA"]

    LOG.debug(f"Writing metadata to pickled file: {fname}")
    df.to_pickle(fname)


def add_metadata_to_pcls_in_toml(ftoml):
    config = toml.load(ftoml)
    data = config["data"]
    keys = ["dscan", "tscan", "bscan"]
    basepath = data["basepath"]
    for key in keys:
        scan = data.get(key)
        if not scan:
            continue
        paths = scan["fnames"]
        force_dx = scan.get("fix_dx")
        for path in paths:
            add_metadata_to_pickled_df(basepath / Path(path), force_dx=force_dx)


def scan_files_from_toml(tom: Union[os.PathLike, dict]) -> tuple:
    try:
        tom = toml.load(tom)
    except TypeError:
        pass

    dscan_paths = _files_from_config(tom, "dscan")
    tscan_paths = _files_from_config(tom, "tscan")

    try:
        bscan_paths = _files_from_config(tom, "bscan")
    except KeyError:
        bscan_paths = None

    return dscan_paths, tscan_paths, bscan_paths


def title_from_toml(tom: Union[os.PathLike, dict]) -> tuple:
    try:
        tom = toml.load(tom)
    except TypeError:
        pass

    try:
        return tom["title"]
    except KeyError:
        return ""



def load_config(fname: os.PathLike) -> SliceEnergySpreadMeasurement:

    config = toml.load(fname)

    oconfig = _optics_config_from_dict(config)

    dscan_paths, tscan_paths, bscan_paths = scan_files_from_toml(config)

    try:
        calib = config["optics"]["tds"]["calibration"]
    except KeyError:
        raise MalformedESMEConfigFile("Missing calibration information")

    try:
        percentages = calib["percentages"]
    except KeyError:
        raise MalformedESMEConfigFile("TDS % info is missing from esme file")

    try:
        screen_dispersion = calib["screen_dispersion"]
    except KeyError:
        raise MalformedESMEConfigFile("Dispersion at which TDS was calibrated is" " missing from esme run file")

    if voltages := calib.get("voltages"):
        calibrator = TrivialTDSCalibrator(percentages, voltages, screen_dispersion)
    else:
        tds_slopes = calib["tds_slopes"]
        tds_slopes_units = calib["tds_slope_units"]
        calibrator = TDSCalibrator(percentages, tds_slopes, screen_dispersion, tds_slope_units=tds_slopes_units)

    dscan = DispersionScan(
        dscan_paths,
        calibrator=calibrator,
    )

    tscan = TDSScan(
        tscan_paths,
        calibrator=calibrator,
    )

    bscan = None
    if bscan_paths:
        bscan = BetaScan(
            bscan_paths,
            calibrator=calibrator,
        )

    return SliceEnergySpreadMeasurement(dscan, tscan, oconfig, bscan=bscan)


def setpoint_snapshots_from_pcls(pcl_files):
    result = []
    for path in pcl_files:
        with path.open("rb") as f:
            result.append(pickle.load(f))
    return result


def make_measurement_runner(name, fconfig, outdir="./", measure_dispersion=False):
    config = toml.load(fconfig)
    LOG.debug(f"Making MeasurementRunner instance from config file: {fconfig}")
    tscan_config = TDSScanConfiguration.from_config_file(fconfig)
    dscan_config = DispersionScanConfiguration.from_config_file(fconfig)

    if measure_dispersion is not None:
        measure_dispersion = make_dispersion_measurer(fconfig)

    return MeasurementRunner(name, dscan_config, tscan_config, outdir=outdir, dispersion_measurer=measure_dispersion)


def make_dispersion_measurer(fconfig):
    config = toml.load(fconfig)
    confd = config["dispersion"]
    a1_voltages = np.linspace(confd["a1_voltage_min"], confd["a1_voltage_max"], num=confd["a1_npoints"])
    return DispersionMeasurer(a1_voltages)


def find_scan_config(fconfig: Path, default_name):
    if fconfig.exists():
        return fconfig
    else:
        return Path(default_name)


def rm_pcl(fpcl: os.PathLike, dry_run=True):
    """Delete a .pcl snapshots file and all images it refers to."""
    fpcl = Path(fpcl)
    pdir = fpcl.resolve().parent

    df = pd.read_pickle(fpcl)

    image_file_names = df[DUMP_SCREEN_ADDRESS]

    for image_name in image_file_names:
        image_path = pdir / image_name
        if dry_run:
            print(f"would rm '{image_path}'")
        else:
            try:
                image_path.unlink()
            except FileNotFoundError as exc:
                LOG.warning(f"{exc.strerror}: {image_path}")

    if dry_run:
        print(f"would rm {fpcl}")
    else:
        fpcl.unlink()


def _loop_pcl_df_files(paths, scan_type):
    for fdf in paths:
        try:
            snapshot = raw_df_pcl_to_setpoint_snapshots(fdf, scan_type)
        except TypeError:
            continue
        else:
            with fdf.open("wb") as f:
                pickle.dump(snapshot, f)


def raw_df_pcl_to_setpoint_snapshots(fpcl, scan_type):
    # try:
    df = pd.read_pickle(fpcl)
    # except EOFError:


    if isinstance(df, SetpointSnapshots):
        LOG.info(f"{fpcl} is already a pickled SetpointSnapshots instance")
        raise TypeError("already a SetpointSnapshot")

    # Measured dispersion
    dx = df["MY_SCREEN_DX"].iloc[0]

    try:
        beta = df["MY_SCREEN_BETA"]
    except KeyError:
        beta = None
    else:
        beta = beta.iloc[0]
        df = df.drop(columns="MY_SCREEN_BETA")

    df = df.drop(columns="MY_SCREEN_DX")

    dispersion_setpoints = np.array([0.6, 0.8, 1.0, 1.2])
    isetpoint = np.abs(dispersion_setpoints - dx).argmin()

    dispersion_setpoint = dispersion_setpoints[isetpoint]

    return SetpointSnapshots(df, scan_type, dispersion_setpoint=dispersion_setpoint,
                             measured_dispersion=(dx, 0.0),
                             beta=beta)



def toml_dfs_to_setpoint_snapshots(ftoml):
    dscan_paths, tscan_paths, bscan_paths = scan_files_from_toml(ftoml)
    if dscan_paths:
        _loop_pcl_df_files(dscan_paths, ScanType.DISPERSION)
    if tscan_paths:
        _loop_pcl_df_files(tscan_paths, ScanType.TDS)
    if bscan_paths:
        _loop_pcl_df_files(bscan_paths, ScanType.BETA)
