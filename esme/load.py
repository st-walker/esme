# #!/usr/bin/env python3

import os
from pathlib import Path
import yaml
import logging
from typing import Optional
import datetime

import numpy as np
import pandas as pd
import toml

from esme.control.scanner import QuadScan, QuadScanSetpoint
from esme.analysis import AnalysisAddresses, OpticsFixedPoints, SetpointDataFrame, MeasurementDataFrames, Scan
from esme.calibration import CompleteCalibration, CalibrationOptics
from esme.core import DiagnosticRegion
from esme.calibration import AmplitudeVoltageMapping


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


def find_scan_config(fconfig: Path, default_name):
    if fconfig.exists():
        return fconfig
    else:
        return Path(default_name)

def i1_dscan_config_from_scan_config_file(config_path: os.PathLike):
    with open(config_path, "r") as f:
        conf = yaml.full_load(f)
    # conf = toml.load(config_path)
    return _dscan_config_from_scan_config(conf, "I1")



def load_result_directory(dirname):
    dirname = Path(dirname)
    scan_setpoint_path = dirname / "scan.yaml"

    with scan_setpoint_path.open("r") as f:
        scan_conf = yaml.safe_load(f)

    image_address = scan_conf["channels"]["image"]
    energy_address = scan_conf["channels"]["energy_at_screen"]

    ofp = OpticsFixedPoints(beta_screen=scan_conf["beta_screen"],
                            beta_tds=scan_conf["beta_tds"],
                            alpha_tds=scan_conf["alpha_tds"])
    files = list(dirname.glob("*m.pkl")) + list(dirname.glob("background.pkl"))
    # Calculate bunch length for maximum streaking case.
    addresses = AnalysisAddresses(image=image_address,
                                  energy=energy_address,
                                  amplitude_sp=scan_conf["channels"]["amplitude_sp"],
                                  amplitude_rb=scan_conf["channels"]["amplitude_rb"],
                                  power_sp=scan_conf["channels"]["power_sp"])

    measurement = load_measurement_dataframes(files, addresses, image_directory=dirname, ofp=ofp)

    return measurement



def get_files_from_directory():
    pass

def load_measurement_dataframes(fnames, addresses, image_directory, ofp):

    dscans = []
    tscans = []
    bscans = []
    bg = 0.0

    # There are two types of background.  One taken before the
    # measurement and one taken before each setpoint of a measurement.
    # background.pkl just is before a measurement, same then used for all
    # otherwise we have background images in the pkl files mixed with the data.

    for i, f in enumerate(fnames):
        bgims = []
        if f.name == "background.pkl":
            bgdf = pd.read_pickle(f)
            image_paths = image_directory / bgdf[addresses.image]
            for path in list(image_paths):
                image = np.load(path)["image"]
                bgims.append(image)
            # Transpose to match convension elsewhere
            bg = np.mean(bgims, axis=0).T

    for f in fnames:
        print(f)
        df = pd.read_pickle(f)
        try:
            df[addresses.image] = image_directory.resolve() / df[addresses.image]
        except:
            LOG.warning(f"Skipping with missing address: {f}")
            continue
            # import ipdb; ipdb.set_trace()
        if "tscan" in str(f):
            tscans.append(SetpointDataFrame(df, addresses=addresses, optics=ofp, bg=bg))
        elif "dscan" in str(f):
            dscans.append(SetpointDataFrame(df, addresses=addresses, optics=ofp, bg=bg))
        elif "bscan" in str(f):
            bscans.append(SetpointDataFrame(df, addresses=addresses, optics=ofp, bg=bg))
        elif f.name == "background.pkl":
            continue
        else:
            raise ValueError(f"Unrecognised file: {f}")

    # from IPython import embed; embed()
    return MeasurementDataFrames(dscans=Scan(dscans),
                                 bscans=Scan(bscans),
                                 tscans=Scan(tscans),
                                 optics=ofp,
                                 bg=bg)

def b2_dscan_config_from_scan_config_file(config_path: os.PathLike):
    conf = toml.load(config_path)
    return _dscan_config_from_scan_config(conf, "b2")


def _dscan_config_from_scan_config(
    dconf: dict, section
) -> QuadScan:

    i1_scan = next(d for d in dconf["scanner"]["scans"] if d["area"] == section)

    setpoints = []
    for dsetpoint in i1_scan["dispersion_scan_setpoints"]:
        setpoints.append(QuadScanSetpoint(k1ls=dsetpoint["k1ls"],
                                          dispersion=dsetpoint["dispersion"],
                                          beta=dsetpoint["beta"]))

    return QuadScan(setpoints, voltage=i1_scan["dispersion_scan_tds_voltage"])


def i1_tscan_config_from_scan_config_file(config_path: os.PathLike):
    return _tscan_config_from_scan_config_file("i1", config_path)

def load_amplitude_voltage_mapping(ftoml: os.PathLike) -> AmplitudeVoltageMapping:
    with Path(ftoml).open("r") as f:
        avmapping_kvps = toml.load(f)
        area = DiagnosticRegion[avmapping_kvps["area"]]
        amplitudes = avmapping_kvps["amplitudes"]
        voltages = avmapping_kvps["voltages"]
        calibration_time = avmapping_kvps.get("calibration_time")
        if calibration_time is not None:
            calibration_time = datetime.datetime.fromtimestamp(calibration_time)
        return AmplitudeVoltageMapping(area, amplitudes, voltages, calibration_time=calibration_time)


def load_calibration_from_yaml(yaml_file_path: str) -> CompleteCalibration:
    with open(yaml_file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)

    region = DiagnosticRegion[yaml_data['area']]
    screen_name = yaml_data['screen']

    optics = CalibrationOptics(
        energy=yaml_data['energy'],
        magnets=yaml_data.get('magnets'),
        r12_streaking=yaml_data.get('r12_streaking'),
        frequency=yaml_data.get('frequency', 3e9)  # Use default if not provided
    )
    amplitudes = yaml_data['amplitudes']
    try:
        cal_factors = yaml_data['cal_factors']
        cal_units = yaml_data['cal_units']
    except KeyError:
        voltages = yaml_data.get("voltages", None)
        voltages = np.array([float(v) for v in voltages])  # Convert to float just in case...
        cal_factors = None
    else:
        cal_scaling = {"m/ps": 1e12, "um/ps": 1e6, "m/s": 1}[cal_units]
        cal_factors = np.array(cal_factors) * cal_scaling
        voltages = None

    # Initialize the CompleteCalibration instance
    complete_calibration = CompleteCalibration(region,
                                               screen_name=screen_name,
                                               optics=optics,
                                               amplitudes=amplitudes,
                                               cal_factors=cal_factors,
                                               voltages=voltages)

    return complete_calibration

def load_calibration_from_result_directory(dirname: os.PathLike, calibration_file_path: Optional[os.PathLike] = None):
    dirname = Path(dirname)
    scan_setpoint_path = dirname / "scan.yaml"

    with scan_setpoint_path.open("r") as f:
        scan_conf = yaml.safe_load(f)

    try:
        calibration_file_path = scan_conf["calibration_file"]
    except KeyError:
        pass
    else:
        return load_calibration_from_yaml(dirname / calibration_file_path)
