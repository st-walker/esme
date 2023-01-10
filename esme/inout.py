#!/usr/bin/env python3

import logging
import re
import os
from pathlib import Path
import contextlib
from typing import Union

import toml
import pandas as pd

from esme.analysis import (DispersionScan, OpticalConfig,
                           SliceEnergySpreadMeasurement, TDSScan, BetaScan, ScanMeasurement)
from esme.calibration import TDSCalibrator, TrivialTDSCalibrator
from esme.injector_channels import TDS_AMPLITUDE_READBACK_ADDRESS

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
    LOG.info(f"Adding metadata to pickled file: {fname}")
    tds_amplitude = _tds_magic_number_from_filename(fname)

    if not force_dx:
        dispersion = _dispersion_from_filename(fname)
    else:
        dispersion = force_dx

    try:
        beta = _beta_from_filename(fname)
    except MissingMetadataInFileNameError:
        beta = None

    df = pd.read_pickle(fname)

    LOG.info(f"Adding dx, tds: {dispersion=}, {tds_amplitude=}")
    df[ScanMeasurement.DF_DX_SCREEN_KEY] = dispersion
    df[TDS_AMPLITUDE_READBACK_ADDRESS] = tds_amplitude

    if beta:
        LOG.info(f"Adding BETA metadata to pickled file: {beta=}")
        df[ScanMeasurement.DF_BETA_SCREEN_KEY] = beta
    else:
        with contextlib.suppress(KeyError):
            del df[ScanMeasurement.DF_BETA_SCREEN_KEY]

    LOG.info(f"Writing metadata to pickled file: {fname}")
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
        bscan = None

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
        raise MalformedESMEConfigFile("Dispersion at which TDS was calibrated is"
                                      " missing from esme run file")

    if voltages := calib.get("voltages"):
        calibrator = TrivialTDSCalibrator(percentages, voltages, screen_dispersion)
    else:
        tds_slopes = calib["tds_slopes"]
        tds_slopes_units = calib["tds_slope_units"]
        calibrator = TDSCalibrator(percentages, tds_slopes,
                                   screen_dispersion,
                                   tds_slope_units=tds_slopes_units)

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
