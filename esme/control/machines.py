from .kickers import FastKickerController
from .dint import DOOCSInterface
from .screens import Screen
from .tds import TransverseDeflector
from .sbunches import SpecialBunchesControl
from .scanner import Scanner
from .optics import MachineLinearOptics
from .exceptions import EuXFELUserError
from esme import DiagnosticRegion


class HighResolutionEnergySpreadMachine:
    """set of high level interfaces to the EuXFEL for measuring the
    slice energy spread with high resolution.

    """
    def __init__(self,
                 *,
                 scanner: Scanner,
                 screen: Screen,
                 deflector: TransverseDeflector,
                 sbunches: SpecialBunchesControl,
                 optics:MachineLinearOptics,
                 di=None) -> None:
        self.di = di if di else DOOCSInterface()
        self.scanner = scanner
        self.screen = screen
        self.deflector = deflector
        self.sbunches = sbunches
        self.optics = optics

    def beam_off(self):
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def beam_on(self):
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def is_beam_on(self):
        return bool(self.di.get_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"))
        


class LPSMachine:
    def __init__(self, region: DiagnosticRegion,
                 kickerop: FastKickerController,
                 screens: dict[str, Screen],
                 tds: TransverseDeflector,
                 optics: MachineLinearOptics,
                 sbunches,
                 di=None) -> None:
        di = di if di else DOOCSInterface()
        self.region = region
        self.kickerop = kickerop
        self.screens = screens
        self.deflector = tds
        self.optics = optics
        self.sbunches = sbunches
        
    def set_kickers_for_screen(self, screen_name: str) -> None:
        screen = self.screens[screen_name]
        try:
            kicker_setpoints = screen.get_fast_kicker_setpoints()
        except EuXFELUserError: # Then there is no kicker info (e.g. dump screen)
            return
        for setpoint in kicker_setpoints:
            self.kickerop.apply_fast_kicker_setpoint(setpoint)
        # self.sbunches.write_kicker

    # def set_use_fast_kickers(self, screen_name):
    #     screen_name = 

# Interface with the EuXFEL limited to what is necessary for the
# effective operation of the special bunch midlayer.
class SpecialBunchLayerInterface:
    def __init__(self, screens, kickerop, di=None):
        self.screens = screens
        self.kickerop = kickerop

    

    
