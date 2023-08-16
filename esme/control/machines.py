from .kickers import FastKickerController
from .mint import XFELMachineInterface
from .screens import ScreenService
from .tds import TransverseDeflectors
from .sbunches import SpecialBunchesControl, DiagnosticRegion

class BunchLengthMachine:
    def __init__(self, kickerop: FastKickerController,
                 sservice: ScreenService,
                 deflectors: TransverseDeflectors,
                 initial_location=None,
                 mi=None) -> None:
        initial_location = initial_location if initial_location else DiagnosticRegion("I1")
        self.kickerop = kickerop
        self.screens = sservice
        self.deflectors = deflectors
        self.sbunches = SpecialBunchesControl()
        self.set_measurement_location(initial_location)
        mi = mi if mi else XFELMachineInterface()
        
    def set_kicker_for_screen(self, screen_name: str) -> None:
        kicker_setpoints = self.screens.get_fast_kicker_setpoints_for_screen(screen_name)
        for setpoint in kicker_setpoints:
            self.kickerop.apply_fast_kicker_setpoint(setpoint)
        # self.sbunches.write_kicker

    # def start_special_bunch(self):
    #     pass

    def set_measurement_location(self, location: DiagnosticRegion):
        self.sbunches.area = location
        self.screens.location = location
        self.deflectors.location = location
    
    # def read_lh_on_state(self):
    #     pass

    # def toggle_lh(self):
    #     pass
