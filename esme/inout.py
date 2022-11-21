#!/usr/bin/env python3

import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import toml

from esme.analysis import DispersionScan, OpticalConfig, SliceEnergySpreadMeasurement, TDSScan


class MalformedESMEConfigFile(RuntimeError):
    pass


def _optics_config_from_dict(config: dict) -> OpticalConfig:
    tds = config["optics"]["tds"]
    screen_betax = config["optics"]["screen"]["betx"]

    return OpticalConfig(
        tds_length=tds["length"],
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

    slopes = config["optics"]["tds"].get("slopes")
    slope_units = config["optics"]["tds"].get("slope_units")
    if slopes and not slope_units:
        raise MalformedESMEConfigFile("TDS slopes provided but without any units...")
    if slopes:
        slopes = np.array(slopes)
        units = {"m/ps": 1e12}
        try:
            scale = units[slope_units]
        except KeyError:
            raise MalformedESMEConfigFile(f"Unknown slope units: {slope_units}")
        else:
            slopes *= scale

    voltages = config["optics"]["tds"].get("voltages")
    dscan = DispersionScan(dscan_paths,
                           tds_slopes=slopes,
                           tds_voltages=voltages,
                           bad_images_per_measurement=config["data"]["dscan"].get("bad_images")
                           )
    tscan = TDSScan(tscan_paths,
                    tds_slopes=slopes,
                    tds_voltages=voltages,
                    bad_images_per_measurement=config["data"]["tscan"].get("bad_images")
                    )

    return SliceEnergySpreadMeasurement(dscan, tscan, oconfig)
