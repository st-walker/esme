from typing import Optional, Union
from enum import Enum, auto
import logging

from .mint import XFELMachineInterface

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class DiagnosticRegion(str, Enum):
    I1 = "I1"
    B2 = "B2"
    UNKNOWN = "UNKNOWN"


class SpecialBunchesControl:
    STATUS_IS_OK_VALUE = 0
    DIAG_BUNCH_FIRE_VALUE = 1
    DIAG_BUNCH_STOP_VALUE = 0
    def __init__(self, location: Optional[DiagnosticRegion] = None,
                 mi: Optional[XFELMachineInterface] = None) -> None:
        self.location = location if location else DiagnosticRegion.UNKNOWN
        self.mi = mi if mi else XFELMachineInterface()

    def set_beam_region(self, br: int) -> None:
        """Note this is zero counting!"""
        ch = self.beamregion_address()
        LOG.info(f"Setting beam region: {ch} = {br}")
        self.mi.set_value(ch, br)

    def get_beam_region(self, br: int) -> None:
        """Note this is zero counting!"""
        ch = self.beamregion_address()
        return self.mi.set_value(ch)

    def set_npulses(self, npulses: int) -> None:
        ch = self.npulses_address()
        LOG.info(f"Setting npulses: {ch} = {npulses}")
        self.mi.set_value(ch, npulses)

    def get_npulses(self) -> int:
        ch = self.npulses_address()
        return self.mi.get_value(ch)

    def get_kicker_name_to_kicker_index_map(self) -> dict[str, int]:
        knumbers = self.mi.get_value("XFEL.SDIAG/SPECIAL_BUNCHES.ML/K*/KICKER_NUMBER")
        rdict = {}
        for kicker_number, *_, kicker_name in knumbers:
            rdict[kicker_name] = kicker_number
        LOG.debug(f"Built kicker name to kicker index map: {rdict}")
        return rdict

    def get_kicker_control_list(self) -> list[int, int, int, int]:
        return self.mi.get_value(self.control_address())

    def set_bunch_number(self, bn) -> None:
        clist = self.get_kicker_control_list()
        clist[0] = int(bn)
        self.mi.set_value(self.control_address(), clist)

    def set_use_tds(self, use_tds: bool) -> None:
        clist = self.get_kicker_control_list()
        clist[1] = int(use_tds)
        self.mi.set_value(self.control_address(), clist)

    def set_kicker(self, kicker_name) -> None:
        clist = self.get_kicker_control_list()
        kmap = self.get_kicker_name_to_kicker_index_map()
        kicker_number = kmap[kicker_name]
        clist[2] = kicker_number
        LOG.info(f"Writing to CONTROL, {kicker_name=}, {kicker_number=}")
        self.mi.set_value(self.control_address(), clist)

    def set_dont_use_fast_kickers(self):
        clist = self.get_kicker_control_list()
        kmap = self.get_kicker_name_to_kicker_index_map()
        kicker_number = kmap[kicker_name]
        clist[2] = kicker_number
        LOG.info(f"Writing to CONTROL, {kicker_name=}, {kicker_number=}")
        self.mi.set_value(self.control_address(), clist)

    def control_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/CONTROL".format(self.location)

    def beamregion_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/SUBTRAIN".format(self.location)

    def status_address(self, thing: str) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/STATUS.{}".format(self.location, thing)

    def npulses_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/PULSES.ACTIVE".format(self.location)

    def fire_diagnostic_bunch_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/START".format(self.location)

    def is_tds_ok(self) -> bool:
        value = self.mi.get_value(self.status_address("TDS"))
        return value == self.STATUS_IS_OK_VALUE

    def is_screen_ok(self) -> bool:
        value = self.mi.get_value(self.status_address("SCREEN"))
        return value == self.STATUS_IS_OK_VALUE

    def is_kicker_ok(self) -> bool:
        value = self.mi.get_value(self.status_address("KICKER"))
        return value == self.STATUS_IS_OK_VALUE

    def is_diag_bunch_firing(self) -> bool:
        ch = self.fire_diagnostic_bunch_address()
        return self.mi.get_value(ch) == self.DIAG_BUNCH_FIRE_VALUE

    def start_diagnostic_bunch(self) -> None:
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Starting diagnostic bunches: {ch} = {self.DIAG_BUNCH_FIRE_VALUE}")
        self.mi.set_value(ch, self.DIAG_BUNCH_FIRE_VALUE)

    def stop_diagnostic_bunch(self) -> None:
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Stopping diagnostic bunches: {ch} = {self.DIAG_BUNCH_STOP_VALUE}")
        self.mi.set_value(ch, self.DIAG_BUNCH_STOP_VALUE)

    # def dump_all(self) -> dict:
    #     addresses = [self.control_address(),
    #                  self.beamregion_address(),
    #                  self.status_address("TDS"),
    #                  self.status_address("SCREEN"),
    #                  self.status_address
                     
                     
