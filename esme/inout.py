#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import toml

from esme.analysis import DispersionScan, OpticalConfig, SliceEnergySpreadMeasurement, TDSScan


class MalformedESMEConfigFile(RuntimeError):
    pass


def _optics_config_from_dict(config: dict) -> OpticalConfig:
    tds = config["optics"]["tds"]
    screen_betax = config["optics"]["screen"]["betx"]

    try:
        tds_voltages = np.array(tds["voltages"])
    except KeyError:
        try:
            tds_voltages = toml.load(tds["crude_calib_file"])
        except FileNotFoundError:
            raise MalformedESMEConfigFile("Missing TDS voltage information")

    return OpticalConfig(
        tds_length=tds["length"],
        tds_voltages=tds_voltages,
        tds_wavenumber=tds["wavenumber"],
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


def load_config(fname: os.PathLike) -> SliceEnergySpreadMeasurement:

    config = toml.load(fname)

    oconfig = _optics_config_from_dict(config)

    # Expand fnames by prepending the provided base path
    dscan_paths = _files_from_config(config, "dscan")
    tscan_paths = _files_from_config(config, "tscan")

    dscan = DispersionScan(dscan_paths)
    tscan = TDSScan(tscan_paths, config["data"]["tscan"].get("bad_images"))

    return SliceEnergySpreadMeasurement(dscan, tscan, oconfig)
