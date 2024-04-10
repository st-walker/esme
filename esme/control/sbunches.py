"""Module for controlling the Special Bunch Midlayer (SBM) for diagnostic bunches

It is responsible for firing the kickers in tandem with the TDS so
that indiividual bunches along a bunch train can be extracted and
observed on off-axis screens.  This module is designed to be an easy
way to set the SBM for the desired use:

1. Select a beam region and bunch number to be kicked, streaked, both or neither.
2. Select a kicker or kickers to fire (or not, if e.g. for an on-axis screen).
3. Select whether to use the TDS in that area (or not if you don't want the beam streaked).

The SBM is generally screen-agnostic and doesn't know much about
screens (although there is some duplication here---see
`SpecialBunchControl.is_screen_ok`).  It assumes that the kicker
voltages and timings are configured correctly if you have a specific
screen in mind.

The SBM is specific to the two diagnostic regions as well, either in
I1 or B2.  It is meaningless to mix these two, e.g. firing the TDS in
the injector alongside kickers in B2.

"""


from typing import Optional
import logging

from .dint import DOOCSInterface

from esme import DiagnosticRegion

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class SpecialBunchesControl:
    STATUS_IS_OK_VALUE = 0
    DIAG_BUNCH_FIRE_VALUE = 1
    DIAG_BUNCH_STOP_VALUE = 0
    DONT_USE_KICKERS_DUMMY_KICKER_NUMBER = 666
    LFF_X_ADDRESS = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/ADAPTIVE_FF_EN_X"
    LFF_Y_ADDRESS = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/ADAPTIVE_FF_EN_Y"

    def __init__(self, location: Optional[DiagnosticRegion] = None,
                 di: Optional[DOOCSInterface] = None) -> None:
        self.location: DiagnosticRegion = location if location else DiagnosticRegion.I1
        self.di = di if di else DOOCSInterface()

    def set_beam_region(self, br: int) -> None:
        """Note is zero counting (weird)!"""
        ch = self.beamregion_address()
        LOG.info(f"Setting beam region (zero counting here): {ch} = {br}")
        self.di.set_value(ch, br)

    def get_beam_region(self) -> None:
        """Note this is zero counting!"""
        ch = self.beamregion_address()
        return self.di.get_value(ch)

    def set_npulses(self, npulses: int) -> None:
        ch = self.npulses_address()
        LOG.info(f"Setting npulses: {ch} = {npulses}")
        self.di.set_value(ch, npulses)

    def get_npulses(self) -> int:
        ch = self.npulses_address()
        return self.di.get_value(ch)

    def get_kicker_name_to_kicker_index_map(self) -> dict[str, int]:
        knumbers = self.di.get_value("XFEL.SDIAG/SPECIAL_BUNCHES.ML/K*/KICKER_NUMBER")
        rdict = {}
        for kicker_number, *_, kicker_name in knumbers:
            rdict[kicker_name] = kicker_number
        LOG.debug(f"Built kicker name to kicker index map: {rdict}")
        return rdict

    def get_kicker_control_list(self) -> list[int, int, int, int]:
        return self.di.get_value(self.control_address())

    def set_bunch_number(self, bn) -> None:
        clist = self.get_kicker_control_list()
        clist[0] = int(bn)
        self.di.set_value(self.control_address(), clist)

    def get_bunch_number(self) -> int:
        clist = self.get_kicker_control_list()
        return int(clist[0])

    def set_use_tds(self, use_tds: bool) -> None:
        clist = self.get_kicker_control_list()
        clist[1] = int(bool(use_tds))
        self.di.set_value(self.control_address(), clist)

    def get_use_tds(self) -> bool:
        clist = self.get_kicker_control_list()
        return bool(clist[1])

    def set_kicker_name(self, kicker_name: str) -> None:
        clist = self.get_kicker_control_list()
        kmap = self.get_kicker_name_to_kicker_index_map()
        kicker_number = kmap[kicker_name]
        clist[2] = kicker_number
        LOG.info(f"Writing to CONTROL, {kicker_name=}, {kicker_number=}")
        self.di.set_value(self.control_address(), clist)

    def get_use_kicker(self):
        value = False
        for readout in self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/*{self.location.name}/KICKER.ON"):
            value, *_, loc = readout
            _, location = loc.split(".")
            if location == self.location:
                break
        ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{self.location.name}/KICKER.ON"
        LOG.debug(f"Read {value} from {ch}")
        return value

    def power_on_kickers(self) -> None:
        # Why is this *?
        *_, loc = self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/*{self.location.name}/KICKER.ON")[0]
        ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{loc}/KICKER.ON"
        value = 0
        LOG.debug(f"Setting value: {ch=} value={value}")
        self.di.set_value(ch, value)

    def power_off_kickers(self) -> None:
        *_, loc = self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/*{self.location.name}/KICKER.ON")
        ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{loc}/KICKER.ON"
        value = 1
        LOG.debug(f"Setting value: {ch=} value={value}")
        self.di.set_value(ch, value)

    def dont_use_kickers(self):
        clist = self.get_kicker_control_list()
        clist[2] = self.DONT_USE_KICKERS_DUMMY_KICKER_NUMBER
        LOG.info(f"Writing to CONTROL, {self.control_address()}, {clist}")
        self.di.set_value(self.control_address(), clist)

    def control_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/CONTROL".format(self.location.name)

    def beamregion_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/SUBTRAIN".format(self.location.name)

    def status_address(self, thing: str) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/STATUS.{}".format(self.location.name, thing)

    def npulses_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/PULSES.ACTIVE".format(self.location.name)

    def fire_diagnostic_bunch_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/START".format(self.location.name)

    def is_tds_ok(self) -> bool:
        value = self.di.get_value(self.status_address("TDS"))
        return value == self.STATUS_IS_OK_VALUE

    def is_screen_ok(self) -> bool:
        value = self.di.get_value(self.status_address("CAMERA"))
        return value == self.STATUS_IS_OK_VALUE

    def is_kicker_ok(self) -> bool:
        value = self.di.get_value(self.status_address("KICKER"))
        return value == self.STATUS_IS_OK_VALUE

    def is_diag_bunch_firing(self) -> bool:
        ch = self.fire_diagnostic_bunch_address()
        return self.di.get_value(ch) == self.DIAG_BUNCH_FIRE_VALUE

    def start_diagnostic_bunch(self) -> None:
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Starting diagnostic bunches: {ch} = {self.DIAG_BUNCH_FIRE_VALUE}")
        self.di.set_value(ch, self.DIAG_BUNCH_FIRE_VALUE)

    def stop_diagnostic_bunch(self) -> None:
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Stopping diagnostic bunches: {ch} = {self.DIAG_BUNCH_STOP_VALUE}")
        self.di.set_value(ch, self.DIAG_BUNCH_STOP_VALUE)

    def ibfb_x_lff_is_on(self) -> bool:
        """Check LFF for IBFB in x-plane"""
        return bool(self.di.get_value(self.LFF_X_ADDRESS))

    def ibfb_y_lff_is_on(self) -> bool:
        """Check LFF for IBFB in x-plane"""
        return bool(self.di.get_value(self.LFF_Y_ADDRESS))

    def is_either_ibfb_on(self) -> bool:
        return self.ibfb_x_lff_is_on() or self.ibfb_y_lff_is_on()

    def set_ibfb_lff(self, *, on: bool) -> None:
        state = int(bool(on))
        self.di.set_value(self.LFF_X_ADDRESS, state)
        self.di.set_value(self.LFF_Y_ADDRESS, state)


    # def dump_all(self) -> dict:
    #     addresses = [self.control_address(),
    #                  self.beamregion_address(),
    #                  self.status_address("TDS"),
    #                  self.status_address("SCREEN"),
    #                  self.status_address
