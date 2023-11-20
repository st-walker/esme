import yaml
import os
from typing import Any
import toml
from pathlib import Path
from collections import defaultdict
from esme.control.tds import StreakingPlane

import pandas as pd

from esme.control.kickers import FastKicker, FastKickerController, PolarityType, FastKickerSetpoint
from esme.control.screens import Screen
from esme.control.machines import HighResolutionEnergySpreadMachine, LPSMachine
from esme.control.tds import TransverseDeflector
from esme.control.sbunches import DiagnosticRegion, SpecialBunchesControl
from esme.control.vdint import ReadBackAddress, ScanMachineInterface, QualifiedImageAddress, WildcardAddress, ReadOnlyDummyAddress, QualifiedImageAddress
from esme.control.scanner import QuadScanSetpoint, ScanConfig, Scanner, QuadScan, TDSScan
from esme.control.snapshot import SnapshotRequest
from esme.control.optics import I1toI1DLinearOptics, I1toB2DLinearOptics
from esme.control.tds import StreakingPlane
from esme import DiagnosticRegion

from esme.analysis import OpticsFixedPoints
from esme.calibration import BolkoCalibrationSetPoint, TDSCalibration, IgorCalibration, DiscreteCalibration, BolkoCalibration, StuartCalibration


def load_kickers_from_config(dconf: dict[str, Any], di=None) -> FastKickerController:
    kicker_defs = dconf["kickers"]
    kickers = []
    for kdef in kicker_defs:
        kickers.append(FastKicker(name=kdef["name"],
                                  adio24_fdl=kdef["adio24_fdl"],
                                  trigger_channel=kdef["trigger_channel"],
                                  di=di))

    return FastKickerController(kickers, di=di)

def load_screens_from_config(dconf: dict[str, Any], di=None) -> dict[str, dict[str, Screen]]:
    screen_defs = dconf["screens"]
    # Mapping of areas (generally I1 or B2 to a dictionary of screen names to screen instances
    screens = defaultdict(lambda: defaultdict(dict))
    for sdef in screen_defs:
        screen_name = sdef["name"]
        location = DiagnosticRegion(sdef["area"])
        try:
            kicker_sp_defs = sdef["kickers"]
        except KeyError:
            kicker_setpoints = None
        else:
            kicker_setpoints = []
            for kicker_name, kicker_sp_def in kicker_sp_defs.items():
                ksp = FastKickerSetpoint(name=kicker_name,
                                         voltage=kicker_sp_def["voltage"],
                                         delay=kicker_sp_def["delay"],
                                         polarity=parse_polarity(kicker_sp_def))
                kicker_setpoints.append(ksp)

        screens[location][screen_name] = Screen(name=screen_name,
                                                fast_kicker_setpoints=kicker_setpoints,
                                                di=di)

    # Convert nested default dicts into plain dicts here for returning
    screens = dict(screens)
    screens = {area: dict(screens[area]) for area in screens}

    return screens

def parse_polarity(cdict):
    try:
        return PolarityType(cdict["polarity"])
    except KeyError:
        return None


# def build_simple_machine_from_config(yamlf: os.PathLike, di=None) -> LPSMachine:
#     with open(yamlf, "r") as f:
#         config = yaml.full_load(f)

#     kickercontroller = load_kickers_from_config(config, di=di)
#     screenservice = load_screens_from_config(config, di=di)
#     deflectors = load_deflectors_from_config(config, di=di)

#     # if di is not None:
#     #     kickercontroller.di = di
#     #     screenservice.di = di
#     #     deflectors.di = di

#     return LPSMachine(kickercontroller, screenservice, deflectors, di=di)


# def build_lps_machine_from_config(yamlf: os.PathLike, di=None) -> LPSMachine:
#     with open(yamlf, "r") as f:
#         config = yaml.full_load(f)

#     scanner = load_scanner_from_config(config, di=di)
#     screenservice = load_screens_from_config(config, di=di)
#     deflectors = load_deflectors_from_config(config, di=di)
#     # optics =

#     return LPSMachine(scanner, screenservice, deflectors, # initial_location=DiagnosticRegion.I1,
#                       di=di)


def build_lps_machine_from_config(yamlf: os.PathLike, area: DiagnosticRegion, di=None):
    with open(yamlf, "r") as f:
        config = yaml.full_load(f)

    all_screens = load_screens_from_config(config, di=di)
    section_screens = all_screens[area]

    kickercontroller = load_kickers_from_config(config, di=di)
    tds = load_deflectors_from_config(config, di=di)[area]

    optics = build_linear_optics(area, di=di)

    sbunches = SpecialBunchesControl(area, di=di)

    return LPSMachine(region=area,
                      kickerop=kickercontroller,
                      screens=section_screens,
                      tds=tds,
                      optics=optics,
                      sbunches=sbunches,
                      di=di)
               

def build_linear_optics(area: DiagnosticRegion, di=None):
    if area is DiagnosticRegion.I1:
        return I1toI1DLinearOptics(di=di)
    elif area is DiagnosticRegion.B2:
        return I1toB2DLinearOptics(di=di)
    else:
        raise ValueError(f"Unrecognised area string: {area}")


def build_b2_lps_machine_from_config(yamlf):
    pass


def load_deflectors_from_config(dconf: dict[str, Any], di=None) -> dict[str, TransverseDeflector]:
    deflector_defs = dconf["deflectors"]

    deflectors = {}
    for ddef in deflector_defs:
        area = ddef["area"]
        sp_fdl = ddef["sp_fdl"]
        rb_fdl = ddef["rb_fdl"]
        plane = StreakingPlane[ddef["streak"].upper()]
        deflectors[str(area)] = TransverseDeflector(sp_fdl, rb_fdl, plane=plane, di=di)

    return deflectors


def load_calibration(fname):
    with open(fname, "r") as f:
        kvps = toml.load(f)

    ctype = kvps["type"]

    if ctype == "bolko":
        return _load_dinimal_bolko_calibration(kvps)
    elif ctype == "igor":
        return _load_igor_calibration(kvps)
    elif ctype == "discrete":
        return _load_discrete_calibration(kvps)
    elif ctype == "stuart":
        return _load_stuart_calibration(kvps)

    raise ValueError("malformed calibration...")


def _load_stuart_calibration(cconf):
    amplitudes = cconf["amplitudes"]
    voltages = cconf["voltages"]
    area = DiagnosticRegion(cconf["area"])
    return StuartCalibration(area, amplitudes, voltages)

def _load_dinimal_bolko_calibration(cconf):
    area = DiagnosticRegion(cconf["area"])
    amplitudes = cconf["amplitudes"]
    slopes = cconf["slopes"]
    r34 = cconf["r34"]
    slope_units = cconf["slope_units"]
    frequency = cconf["frequency"]
    energy = cconf["energy"]

    calibs = []
    for ampl, slope in zip(amplitudes, slopes):
        if slope_units == "um/ps":
            slope *= 1e6
        calibs.append(BolkoCalibrationSetPoint(ampl, slope, r34=r34,
                                       energy=energy,
                                       frequency=frequency))

    return BolkoCalibration(area, calibs)

def _load_igor_calibration(dcalib):
    area = dcalib["area"]
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    return IgorCalibration(amplitudes, voltages)

def _load_discrete_calibration(dcalib):
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    return DiscreteCalibration(amplitudes, voltages)


def load_scanner_from_config(dconf, di=None):
    scans = []
    scanners = []
    for scan in dconf["scanner"]["scans"]:
        quad_setpoints = []
        for dsetpoint in scan["dispersion_scan_setpoints"]:
            dispersion = dsetpoint["dispersion"]
            beta = dsetpoint["beta"]
            k1ls = dsetpoint["k1ls"]
            setpoint = QuadScanSetpoint(k1ls, dispersion=dispersion, beta=beta)
            quad_setpoints.append(setpoint)

        beta_scan_setpoints = []
        for bsetpoint in scan["beta_scan_setpoints"]:
            dispersion = bsetpoint["dispersion"]
            beta = bsetpoint["beta"]
            k1ls = bsetpoint["k1ls"]
            setpoint = QuadScanSetpoint(k1ls, dispersion=dispersion, beta=beta)
            beta_scan_setpoints.append(setpoint)

        channels = scan["channels"]
        for channel in channels:
            snapshot_request = SnapshotRequest(image=channels["image"],
                                               addresses=channels["addresses"],
                                               wildcards=channels["wildcards"])

        tscan_d = scan["tds_scan_dispersion"]
        tds_scan_setpoint = next(sp for sp in quad_setpoints if sp.dispersion == tscan_d)

        tds_scan = TDSScan(scan["tds_scan_voltages"], tds_scan_setpoint)
        quad_scan = QuadScan(quad_setpoints, scan["dispersion_scan_tds_voltage"])
        beta_scan = QuadScan(beta_scan_setpoints, scan["dispersion_scan_tds_voltage"])

        ofp = OpticsFixedPoints(scan["beta_screen"],
                                scan["beta_tds"],
                                scan["alpha_tds"])

        scanconf = ScanConfig(scan["name"],
                              qscan=quad_scan,
                              tscan=tds_scan,
                              bscan=beta_scan,
                              area=DiagnosticRegion(scan["area"]),
                              optics_fixed_points=ofp,
                              screen=scan["screen"],
                              request=snapshot_request
                              )

        scans.append(scanconf)

    return Scanner(scans[0], di=di)

def load_virtual_machine_interface(dconf):
    state = dconf["simple"]

    readbacks = dconf["readbacks"]

    for readback in readbacks:
        state[readback["address"]] = ReadBackAddress(readback["rb"], readback["noise"])

    for readback in readbacks:
        state[readback["address"]] = ReadBackAddress(readback["rb"], readback["noise"])

    images = dconf["images"]
    for image in images:
        address = image["address"]
        image_type = image["type"]

        if image_type == "qualified":
            snapshots_db = pd.read_pickle(image["snapshots_db"])
            filters = image["filters"]
            images_dir = Path(image["images_dir"])
            state[address] = QualifiedImageAddress(address, filters, snapshots_db, images_dir)

        elif image_type == "simple":
            state[address] = pd.read_pickle(image["filename"])

    readonly = dconf["read_only"]
    for address in readonly["addresses"]:
        state[address] = ReadOnlyDummyAddress(address)

    for wildcard in readonly["wildcards"]:
        state[wildcard] = WildcardAddress(wildcard, snapshots_db)

    return ScanMachineInterface(state)

def get_scan_config_for_area(dconf: dict, area: str):
    scans = dconf["scanner"]["scans"]
    for scan in scans:
        if scan["area"] == area:
            return scan

    raise ValueError(f"Unable to find scan information for area: {area}")
