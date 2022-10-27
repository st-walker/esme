#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import toml

from esme.analysis import DispersionScan, OpticalConfig, SliceEnergySpreadMeasurement, TDSScan


class MalformedESMEConfigFile(RuntimeError):
    pass


def _optics_config_from_dict(config: dict) -> OpticalConfig:
    tds_mask = _get_masks(config, "tscan")

    tds = config["optics"]["tds"]
    screen_betax = config["optics"]["screen"]["betx"]

    tds_voltages = np.array(tds["voltages"])[tds_mask]

    return OpticalConfig(
        tds_length=tds["length"],
        tds_voltages=tds_voltages,
        tds_wavenumber=tds["wavenumber"],
        tds_bety=tds["bety"],
        tds_alfy=tds["alfy"],
        ocr_betx=screen_betax,
    )


def _get_masks(config: dict, scan_name: str) -> list[bool]:
    try:
        return config["data"][scan_name]["mask"]
    except KeyError:
        raise MalformedESMEConfigFile("Missing TDS Scan file mask")


def _apply_mask(iterable: Iterable[Any], mask: Iterable[bool]) -> list[Any]:
    return [i for (i, m) in zip(iterable, mask) if m]


def _files_from_config(config, scan_name) -> list[Path]:
    try:
        basepath = Path(config["data"]["basepath"])
    except KeyError:
        basepath = Path(".")

    mask = _get_masks(config, scan_name)

    fnames = config["data"][scan_name]["fnames"]
    paths = [basepath / f for f in fnames]
    paths = _apply_mask(paths, mask)

    return paths

def load_config(fname: os.PathLike) -> SliceEnergySpreadMeasurement:

    config = toml.load(fname)

    oconfig = _optics_config_from_dict(config)

    # Expand fnames by prepending the provided base path
    dscan_paths = _files_from_config(config, "dscan")
    tscan_paths = _files_from_config(config, "tscan")

    dscan = DispersionScan(dscan_paths)
    tscan = TDSScan(tscan_paths)

    return SliceEnergySpreadMeasurement(dscan, tscan, oconfig)
