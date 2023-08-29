import yaml
import os
from typing import Any
import toml
from pathlib import Path

import pandas as pd

from esme.control.kickers import FastKicker, FastKickerController, PolarityType, FastKickerSetpoint
from esme.control.screens import Screen, ScreenService
from esme.control.machines import BunchLengthMachine, LPSMachine
from esme.control.tds import TransverseDeflector, TransverseDeflectors
from esme.control.sbunches import DiagnosticRegion
from esme.control.vmint import ReadBackAddress, ScanMachineInterface, QualifiedImageAddress, WildcardAddress, ReadOnlyDummyAddress, QualifiedImageAddress
from esme.control.scanner import QuadScanSetpoint, ScanConfig, Scanner, QuadScan, TDSScan
from esme.control.snapshot import SnapshotRequest

from esme.analysis import OpticsFixedPoints
from esme.calibration import BolkoCalibrationSetPoint, TDSCalibration, IgorCalibration, DiscreteCalibration


def load_kickers_from_config(dconf: dict[str, Any]) -> FastKickerController:
    kicker_defs = dconf["kickers"]
    kickers = []
    for kdef in kicker_defs:
        kickers.append(FastKicker(name=kdef["name"],
                                  adio24_fdl=kdef["adio24_fdl"],
                                  trigger_channel=kdef["trigger_channel"]))

    return FastKickerController(kickers)

def load_screens_from_config(dconf: dict[str, Any], mi=None) -> ScreenService:
    screen_defs = dconf["screens"]
    screens = []
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

        screens.append(Screen(name=screen_name,
                              location=location,
                              fast_kicker_setpoints=kicker_setpoints,
                              mi=mi))

    return ScreenService(screens, mi=mi)

def parse_polarity(cdict):
    try:
        return PolarityType(cdict["polarity"])
    except KeyError:
        return None


def build_simple_machine_from_config(yamlf: os.PathLike, mi=None) -> BunchLengthMachine:
    with open(yamlf, "r") as f:
        config = yaml.full_load(f)

    kickercontroller = load_kickers_from_config(config)
    screenservice = load_screens_from_config(config)
    deflectors = load_deflectors_from_config(config, mi=mi)

    if mi is not None:
        kickercontroller.mi = mi
        screenservice.mi = mi
        deflectors.mi = mi

    return BunchLengthMachine(kickercontroller, screenservice, deflectors, initial_location=DiagnosticRegion.I1, mi=mi)


def build_lps_machine_from_config(yamlf: os.PathLike, mi=None) -> BunchLengthMachine:
    with open(yamlf, "r") as f:
        config = yaml.full_load(f)

    scanner = load_scanner_from_config(config, mi=mi)
    screenservice = load_screens_from_config(config, mi=mi)
    deflectors = load_deflectors_from_config(config, mi=mi)

    return LPSMachine(scanner, screenservice, deflectors, # initial_location=DiagnosticRegion.I1,
                      mi=mi)


def load_deflectors_from_config(dconf: dict[str, Any], mi=None) -> TransverseDeflectors:
    deflector_defs = dconf["deflectors"]

    deflectors = []
    for ddef in deflector_defs:
        area = DiagnosticRegion(ddef["area"])
        sp_fdl = ddef["sp_fdl"]
        rb_fdl = ddef["rb_fdl"]

        deflector = TransverseDeflector(area, sp_fdl, rb_fdl, mi=mi)
        deflectors.append(deflector)

    return TransverseDeflectors(deflectors)


def load_calibration(fname):
    with open(fname, "r") as f:
        kvps = toml.load(f)

    ctype = kvps["type"]

    if ctype == "bolko":
        return _load_minimal_bolko_calibration(kvps)
    elif ctype == "igor":
        return _load_igor_calibration(kvps)
    elif ctype == "discrete":
        return _load_discrete_calibration(kvps)        

    raise ValueError("malformed calibration...")


def _load_minimal_bolko_calibration(cconf):
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

    return TDSCalibration(calibs)

def _load_igor_calibration(dcalib):
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    return IgorCalibration(amplitudes, voltages)

def _load_discrete_calibration(dcalib):
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    return DiscreteCalibration(amplitudes, voltages)


def load_scanner_from_config(dconf, mi=None):
    scans = []
    scanners = []
    for scan in dconf["scans"]:
        quad_setpoints = []
        for dsetpoint in scan["setpoints"]:
            dispersion = dsetpoint["dispersion"]
            k1ls = dsetpoint["k1ls"]
            setpoint = QuadScanSetpoint(k1ls, dispersion)
            quad_setpoints.append(setpoint)

        channels = scan["channels"]
        for channel in channels:
            snapshot_request = SnapshotRequest(image=channels["image"],
                                               addresses=channels["addresses"],
                                               wildcards=channels["wildcards"])

        tscan_d = scan["tds_scan_dispersion"]
        tds_scan_setpoint = next(sp for sp in quad_setpoints if sp.dispersion == tscan_d)

        tds_scan = TDSScan(scan["tds_scan_voltages"], tds_scan_setpoint)
        quad_scan = QuadScan(quad_setpoints, scan["dispersion_scan_tds_voltage"])

        ofp = OpticsFixedPoints(scan["beta_screen"],
                                scan["beta_tds"],
                                scan["alpha_tds"])

        scanconf = ScanConfig(scan["name"],
                              quad_scan,
                              tds_scan,
                              DiagnosticRegion(scan["area"]),
                              optics_fixed_points=ofp,
                              screen=scan["screen"],
                              request=snapshot_request
                              )
                   
        scans.append(scanconf)

    return Scanner(scans[0], mi=mi)

    # return scans


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
        snapshots_db = pd.read_pickle(image["snapshots_db"])
        filters = image["filters"]
        images_dir = Path(image["images_dir"])
        state[address] = QualifiedImageAddress(address, filters, snapshots_db, images_dir)

    readonly = dconf["read_only"]
    for address in readonly["addresses"]:
        state[address] = ReadOnlyDummyAddress(address)

    for wildcard in readonly["wildcards"]:
        state[wildcard] = WildcardAddress(wildcard, snapshots_db)

    return ScanMachineInterface(state)
