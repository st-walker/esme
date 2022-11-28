#!/usr/bin/env python3

import os
from pathlib import Path

import toml

from esme.analysis import DispersionScan, OpticalConfig, SliceEnergySpreadMeasurement, TDSScan
from esme.calibration import TDSCalibrator, TrivialTDSCalibrator


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

    try:
        calib = config["optics"]["tds"]["calibration"]
    except KeyError:
        raise MalformedESMEConfigFile("Missing calibration information")

    try:
        percentages = calib["percentages"]
    except KeyError:
        raise MalformedESMEConfigFile("TDS % info is missing from esme file")

    if voltages := calib.get("voltages"):
        calibrator = TrivialTDSCalibrator(percentages, voltages)
    else:
        tds_slopes = calib["tds_slopes"]
        tds_slopes_units = calib["tds_slope_units"]
        screen_dispersion = calib["screen_dispersion"]
        calibrator = TDSCalibrator(percentages, tds_slopes,
                                   screen_dispersion,
                                   tds_slope_units=tds_slopes_units)

    dscan = DispersionScan(
        dscan_paths,
        calibrator=calibrator,
        bad_images_per_measurement=config["data"]["dscan"].get("bad_images"),
    )
    tscan = TDSScan(
        tscan_paths,
        calibrator=calibrator,
        bad_images_per_measurement=config["data"]["tscan"].get("bad_images"),
    )

    return SliceEnergySpreadMeasurement(dscan, tscan, oconfig)
