import yaml
import os
from typing import Any, Callable
import toml
from pathlib import Path
from collections import defaultdict
from copy import deepcopy

import pandas as pd

from esme.control.kickers import FastKicker, FastKickerController, FastKickerSetpoint, PolarityType
from esme.control.screens import Screen
from esme.control.machines import HighResolutionEnergySpreadMachine, LPSMachine, MachineManager, DiagnosticBunchesManager
from esme.control.tds import TransverseDeflector, StreakingPlane
from esme.control.sbunches import SpecialBunchesControl
from esme.control.dint import DOOCSInterfaceABC
from esme.control.vdint import (ReadBackAddress,
                               ScanMachineInterface,
                               QualifiedImageAddress,
                               WildcardAddress,
                               ReadOnlyDummyAddress,
                               QualifiedImageAddress)
from esme.control.scanner import QuadScanSetpoint, ScanConfig, Scanner, QuadScan, TDSScan
from esme.control.snapshot import SnapshotRequest
from esme.control.optics import I1toI1DLinearOptics, I1toB2DLinearOptics
from esme.control.tds import StreakingPlane
from esme.core import DiagnosticRegion
from esme.control.mstate import AreaWatcher
from esme.analysis import OpticsFixedPoints


def load_kickers_from_config(dconf: dict[str, Any], area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> FastKickerController:
    kicker_defs = dconf["kickers"]
    kickers = []
    for kdef in kicker_defs:
        kickers.append(FastKicker(name=kdef["name"],
                                  adio24_fdl=kdef["adio24_fdl"],
                                  trigger_channel=kdef["trigger_channel"],
                                  di=di))

    return FastKickerController(kickers, di=di)


def load_screens_from_config(dconf: dict[str, Any], area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> dict[str, Screen]:
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

    return screens[area]

def parse_polarity(cdict: dict[str, Any]) -> PolarityType:
    try:
        return PolarityType(cdict["polarity"])
    except KeyError:
        return None

class MachineManagerFactory:
    def __init__(self, yamlf: os.PathLike, default_dint: DOOCSInterfaceABC | None = None):
        with open(yamlf, "r") as f:
            self._config = yaml.full_load(f)
            self._facade_cache = defaultdict(dict)
            self._manager_cache = defaultdict(dict)
            self._default_di = default_dint or DOOCSInterface()

    def _get_else_build_from_config(self, name: str, area: DiagnosticRegion, backup_fn: Callable[[dict[str, Any], DiagnosticRegion, DOOCSInterfaceABC], MachineManager]):
        try:
            inst = self._facade_cache[area][name]
        except KeyError:
            inst = backup_fn(self._config, area, di=self._default_di)
            self._facade_cache[area][name] = inst
        return inst

    def _get_else_build(self, name: str, area: DiagnosticRegion, backup_fn): # -> SubsystemFacade
        try:
            inst = self._facade_cache[area][name]
        except KeyError:
            inst = backup_fn(area)
            self._facade_cache[area][name] = inst
        return inst

    def _get_screens(self, area: DiagnosticRegion) -> dict[str, Screen]:
        return self._get_else_build_from_config("screens", area, load_screens_from_config)
    
    def _get_kicker_facade(self, area: DiagnosticRegion) -> FastKickerController:
        return self._get_else_build_from_config("kickers", area, load_kickers_from_config)

    def _get_deflector(self, area: DiagnosticRegion) -> dict[str, TransverseDeflector]:
        return self._get_else_build_from_config("deflectors", area, load_deflector_from_config)
    
    def _get_optics(self, area: DiagnosticRegion) -> I1toI1DLinearOptics | I1toB2DLinearOptics:
        return self._get_else_build("optics", area, build_linear_optics)
    
    def _get_sbunches(self, area: DiagnosticRegion) -> SpecialBunchesControl:
        return self._get_else_build("sbunches", area, SpecialBunchesControl)
    
    def _get_scanner(self, area: DiagnosticRegion) -> Scanner:
        return self._get_else_build_from_config("scanner", area, load_scanner_from_config)

    def make_diagnostic_bunches_manager(self, area: DiagnosticRegion) -> DiagnosticBunchesManager:
        try:
            manager = deepcopy(self._manager_cache[area]["diagbunches"])
        except KeyError:
            manager = DiagnosticBunchesManager(screens=self._get_screens(area),
                                               kickerop=self._get_kicker_facade(area),
                                               sbunches=self._get_sbunches(area))
            self._manager_cache[area]["diagbunches"] = manager
        return manager
            
    def make_hires_injector_energy_spread_manager(self) -> HighResolutionEnergySpreadMachine:
        area = DiagnosticRegion.I1
        try:
            manager = deepcopy(self._manager_cache[area]["hires"])
        except KeyError:
            manager = HighResolutionEnergySpreadMachine(scanner=self._get_scanner(area),
                                                        screen=self._get_screens(area)["OTRC.64.I1D"],
                                                        deflector=self._get_deflector(area),
                                                        sbunches=self._get_sbunches(area),
                                                        optics=self._get_optics(area),
                                                        di=self._default_di)
            self._manager_cache[area]["hires"] = manager
        return manager

    def make_lps_manager(self, area: DiagnosticRegion) -> LPSMachine:
        try:
            manager = deepcopy(self._manager_cache[area]["lps"])
        except KeyError:
            manager = LPSMachine(region=area,
                                 kickerop=self._get_kicker_facade(area),
                                 screens=self._get_screens(area),
                                 tds=self._get_deflector(area),
                                 optics=self._get_optics(area),
                                 sbunches=self._get_sbunches(area),
                                 di=self._default_di)
        else:
            self._manager_cache[area]["lps"] = manager
        
        return manager
    
    def make_i1_b2_managers(self) -> tuple[LPSMachine, LPSMachine]:
        return self.make_lps_manager(DiagnosticRegion.I1), self.make_lps_manager(DiagnosticRegion.B2)

        


# def build_lps_machine_from_config(yamlf: os.PathLike, area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> LPSMachine:
#     with open(yamlf, "r") as f:
#         config = yaml.full_load(f)

#     all_screens = load_screens_from_config(config, di=di)
#     section_screens = all_screens[area]

#     kickercontroller = load_kickers_from_config(config, di=di)
#     tds = load_deflectors_from_config(config, di=di)[area]
#     optics = build_linear_optics(area, di=di)
#     sbunches = SpecialBunchesControl(area, di=di)

#     return LPSMachine(region=area,
#                       kickerop=kickercontroller,
#                       screens=section_screens,
#                       tds=tds,
#                       optics=optics,
#                       sbunches=sbunches,
#                       di=di)

def build_area_watcher_from_config(yamlf: os.PathLike, area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> AreaWatcher:
    with open(yamlf, "r") as f:
        config = yaml.full_load(f)

    section_screens = load_screens_from_config(config, area, di=di)

    kickercontroller = load_kickers_from_config(config, area, di=di)
    tds = load_deflector_from_config(config, area, di=di)
    # optics = build_linear_optics(area, di=di)
    sbunches = SpecialBunchesControl(area, di=di)

    return AreaWatcher(kickerop=kickercontroller,
                       screens=section_screens,
                       tds=tds,
 #                      optics=optics,
                       sbunches=sbunches,
                       di=di)
               

def build_linear_optics(area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> I1toI1DLinearOptics | I1toB2DLinearOptics:
    if area is DiagnosticRegion.I1:
        return I1toI1DLinearOptics(di=di)
    elif area is DiagnosticRegion.B2:
        return I1toB2DLinearOptics(di=di)
    else:
        raise ValueError(f"Unrecognised area string: {area}")


def build_b2_lps_machine_from_config(yamlf):
    pass


def load_deflector_from_config(dconf: dict[str, Any], area: DiagnosticRegion, di: DOOCSInterfaceABC | None = None) -> TransverseDeflector:
    deflector_defs = dconf["deflectors"]

    deflectors = {}
    for ddef in deflector_defs:
        area = ddef["area"]
        sp_fdl = ddef["sp_fdl"]
        rb_fdl = ddef["rb_fdl"]
        plane = StreakingPlane[ddef["streak"].upper()]
        deflectors[area] = TransverseDeflector(sp_fdl, rb_fdl, plane=plane, di=di)
    return deflectors[area]


def load_calibration(fname: os.PathLike): # -> ???
    with open(fname, "r") as f:
        kvps = toml.load(f)

    ctype = kvps["type"]

    if ctype == "bolko":
        return _load_minimal_bolko_calibration(kvps)
    elif ctype == "igor":
        return _load_igor_calibration(kvps)
    elif ctype == "discrete":
        return _load_discrete_calibration(kvps)
    elif ctype == "stuart":
        return _load_stuart_calibration(kvps)

    raise ValueError("malformed calibration...")


# def _load_stuart_calibration(cconf):
#     amplitudes = cconf["amplitudes"]
#     voltages = cconf["voltages"]
#     area = DiagnosticRegion(cconf["area"])
#     return StuartCalibration(area, amplitudes, voltages)

# def _load_dinimal_bolko_calibration(cconf):
#     area = DiagnosticRegion(cconf["area"])
#     amplitudes = cconf["amplitudes"]
#     slopes = cconf["slopes"]
#     r34 = cconf["r34"]
#     slope_units = cconf["slope_units"]
#     frequency = cconf["frequency"]
#     energy = cconf["energy"]

#     calibs = []
#     for ampl, slope in zip(amplitudes, slopes):
#         if slope_units == "um/ps":
#             slope *= 1e6
#         calibs.append(BolkoCalibrationSetpoint(ampl, slope, r34=r34,
#                                                energy=energy,
#                                                frequency=frequency))

#     return BolkoCalibration(area, calibs)

def _load_igor_calibration(dcalib: dict[str, Any]): #-> IgorCalibration:
    dcalib["area"]
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    region = DiagnosticRegion(dcalib["area"])
    return IgorCalibration(region, amplitudes, voltages)

def _load_discrete_calibration(dcalib: dict[str, any]): # -> DiscreteCalibration:
    amplitudes = dcalib["amplitudes"]
    voltages = dcalib["voltages"]
    return DiscreteCalibration(amplitudes, voltages)


def load_scanner_from_config(dconf, di: DOOCSInterfaceABC | None = None) -> Scanner:
    scans = []
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

def load_virtual_machine_interface(dconf: dict[str, Any]) -> ScanMachineInterface:
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

def get_scan_config_for_area(dconf: dict[str, Any], area: str) -> dict[str, Any]:
    scans = dconf["scanner"]["scans"]
    for scan in scans:
        if scan["area"] == area:
            return scan

    raise ValueError(f"Unable to find scan information for area: {area}")
