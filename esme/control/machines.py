from .kickers import FastKickerController
from .mint import XFELMachineInterface
from .screens import ScreenService
from .tds import TransverseDeflectors
from .sbunches import SpecialBunchesControl, DiagnosticRegion
from .scanner import Scanner

class BunchLengthMachine:
    def __init__(self, kickerop: FastKickerController,
                 sservice: ScreenService,
                 deflectors: TransverseDeflectors,
                 initial_location=None,
                 mi=None) -> None:
        initial_location = initial_location if initial_location else DiagnosticRegion("I1")
        mi = mi if mi else XFELMachineInterface()
        self.kickerop = kickerop
        self.screens = sservice
        self.deflectors = deflectors
        self.sbunches = SpecialBunchesControl(mi=mi)
        self.set_measurement_location(initial_location)

        
    def set_kicker_for_screen(self, screen_name: str) -> None:
        kicker_setpoints = self.screens.get_fast_kicker_setpoints_for_screen(screen_name)
        for setpoint in kicker_setpoints:
            self.kickerop.apply_fast_kicker_setpoint(setpoint)
        # self.sbunches.write_kicker

    def set_measurement_location(self, location: DiagnosticRegion):
        self.sbunches.location = location
        self.screens.location = location
        self.deflectors.location = location


class LPSMachine:
    def __init__(self, scanner: Scanner,
                 sservice: ScreenService,
                 deflectors: TransverseDeflectors,
                 # initial_location=None,
                 mi=None) -> None:
        # initial_location = initial_location if initial_location else DiagnosticRegion("I1")
        self.mi = mi if mi else XFELMachineInterface()
        self.scanner = scanner
        self.screens = sservice
        self.deflectors = deflectors
        self.sbunches = SpecialBunchesControl(mi=mi)
        # self.set_measurement_location(initial_location)

    def beam_off(self):
        self.mi.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def beam_on(self):
        self.mi.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

        
    # def set_kicker_for_screen(self, screen_name: str) -> None:
    #     kicker_setpoints = self.screens.get_fast_kicker_setpoints_for_screen(screen_name)
    #     for setpoint in kicker_setpoints:
    #         self.kickerop.apply_fast_kicker_setpoint(setpoint)
    #     # self.sbunches.write_kicker

    # def set_measurement_location(self, location: DiagnosticRegion):
    #     self.sbunches.location = location
    #     self.screens.location = location
    #     self.deflectors.location = location

        
