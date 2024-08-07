from dataclasses import dataclass
import logging

import numpy as np

from esme.control.sbunches import DiagnosticRegion
from esme.control.exceptions import EuXFELMachineError
from esme.analysis import OpticsFixedPoints
from esme.control.snapshot import SnapshotRequest, Snapshotter
from esme.control.dint import DOOCSInterface

LOG = logging.getLogger(__name__)

@dataclass
class QuadScanSetpoint:
    k1ls: dict[str, float]
    dispersion: float
    beta: float

    def quad_names(self) -> list[str]:
        return list(self.k1ls.keys())

@dataclass
class QuadScan:
    setpoints: list[QuadScanSetpoint]
    voltage: float

    @property
    def dispersions(self) -> list[float]:
        return [setpoint.dispersion for setpoint in self.setpoints]

    @property
    def betas(self) -> list[float]:
        return [setpoint.beta for setpoint in self.setpoints]

@dataclass
class TDSScan:
    voltages: list[float]
    setpoint: QuadScanSetpoint


@dataclass
class ScanConfig:
    name: str
    qscan: QuadScan
    tscan: TDSScan
    bscan: QuadScan
    area: DiagnosticRegion
    request: SnapshotRequest
    optics_fixed_points: OpticsFixedPoints
    
    screen: str = "OTRC.64.I1D"
    

    # @property
    # def dispersions(self) -> list[float]:
    #     return [setpoint.dispersion for setpoint in self.qscan.setpoints]

    # @property
    # def voltages(self) -> list[float]:
    #     return return  self.tscan.voltages]
    

class ScanSetpointError(EuXFELMachineError):
    pass


class Scanner:
    # TODO: should this use sample address as well or instead?  We are using setpoint..
    FDP_QUAD_KICK_SP_ADDRESS = "XFEL.MAGNETS/MAGNET.ML/{}/KICK_MRAD.SP"
    BEAM_ALLOWED_ADDRESS = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"
    def __init__(self, scan: ScanConfig, di=None):
        self.scan = scan
        self.di = di if di else DOOCSInterface()

    def set_quad_strength(self, quad_name: str, kmrad: float) -> None:
        ch = self.FDP_QUAD_KICK_SP_ADDRESS.format(quad_name)
        LOG.debug(f"Setting quad strength: {ch} = {kmrad}")
        self.di.set_value(ch, kmrad)

    def get_quad_strength(self, quad_name: str) -> float:
        ch = self.FDP_QUAD_KICK_SP_ADDRESS.format(quad_name)
        return self.di.get_value(ch)

    def get_setpoint(self, dispersion: float, beta=None) -> None:
        dispersions = self.scan.qscan.dispersions
        # from IPython import embed; embed()
        index = dispersions.index(dispersion)
        return self.scan.qscan.setpoints[index]

    def set_scan_setpoint_quads(self, setpoint: QuadScanSetpoint) -> None:
        LOG.info(f"Setting setpoint for dispersion: {setpoint.dispersion}")
        # setpoint = self.get_setpoint(dispersion)
        for quad_name, k1l in setpoint.k1ls.items():
            self.set_quad_strength(quad_name, k1l)

    def beam_off(self) -> None:
        self.di.set_value(self.BEAM_ALLOWED_ADDRESS, 0)

    def beam_on(self) -> None:
        self.di.set_value(self.BEAM_ALLOWED_ADDRESS, 1)
        
    def infer_intended_dispersion_setpoint(self) -> QuadScanSetpoint:
        for setpoint in self.scan.setpoints:
            setpoint.dispersion
            for name, k1l in setpoint.k1ls.items():
                actual_k1l = self.get_quad_strength(name)
                if not np.isclose(k1l, actual_k1l):
                    break
            else:
                return setpoint

        raise ScanSetpointError("Could not determine the current scan setpoint in the machine")

    def get_snapshotter(self) -> Snapshotter:
        return Snapshotter(self.scan.request, di=self.di)

    def cycle_scan_quads(self) -> None:
        quad_names = set()
        for setpoint in self.scan.qscan.setpoints:
            quad_names |= set(setpoint.k1ls)
        for setpoint in self.scan.bscan.setpoints:
            quad_names |= set(setpoint.k1ls)
        quad_names |= set(self.scan.tscan.setpoint.k1ls)

        for quad_name in quad_names:
            LOG.info(f"Cycling quadrupole: {quad_name}")
            self.di.set_value(f"XFEL.MAGNETS/MAGNET.ML/{quad_name}/CYCLE", 1)
        

class ScannerConfigWidget:
    pass
