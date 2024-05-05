from typing import Any

from .kickers import FastKickerController
from .dint import DOOCSInterface
from .screens import Screen
from .tds import TransverseDeflector
from .sbunches import SpecialBunchesControl
from .scanner import Scanner
from .optics import MachineLinearOptics
from .exceptions import EuXFELUserError
from esme.core import DiagnosticRegion
from esme.control.snapshot import SnapshotRequest, Snapshotter


# class StreakingPlaneCalibrationMixin:
#     def get_streaking_com_slope(self, screen_name):
#         voltage = self.deflector.get_voltage_rb()
#         r12 = self.machine.optics.r12_streaking_from_tds_to_point(self.name)
#         energy_mev = self.machine.optics.get_beam_energy() # MeV is fine here.
#         com_slope = get_tds_com_slope(r12, energy_mev, voltage)
#         scale_factor = 1 / com_slope
#         return 
    
class MachineManager:
    pass

class HighResolutionEnergySpreadMachine(MachineManager):
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
        

# What is the point in this class besides bringing other bits together?  Does this really need sbunches etc?
class LPSMachine(MachineManager):
    def __init__(self, region: DiagnosticRegion,
                 kickerop: FastKickerController,
                 screens: dict[str, Screen],
                 tds: TransverseDeflector,
                 optics: MachineLinearOptics,
                 sbunches,
                 di=None) -> None:
        di = di or DOOCSInterface()
        self.region = region
        self.kickerop = kickerop
        self.screens = screens
        self.deflector = tds
        self.optics = optics
        self.sbunches = sbunches
        
# Interface with the EuXFEL limited to what is necessary for the
# effective operation of the special bunch midlayer.
class DiagnosticBunchesManager(MachineManager):
    """Minimal collection of instances for interacting with the Special Bunch Midlayer effectively.
    The functionality is basically: Interact with the SBM and also set the kickers for a given screen
    by name and set the SBM to activate the kickers for said screen.
    """
    def __init__(self, *, screens: dict[str, Screen], kickerop: FastKickerController, sbunches: SpecialBunchesControl, di=None):
        di = di or DOOCSInterface()        
        self.screens = screens
        self.kickerop = kickerop
        self.sbunches = sbunches
        
    def set_kickers_for_screen(self, screen_name: str) -> None:
        """For the given screen, set the kicker timings, voltages, etc. 
        so that when the kickers are fired, they will kick the diagnostic bunch
        onto the named screen (screen_name)."""
        # Get the screen instance by name
        screen = self.screens[screen_name]
        # Try and set the kickers for the screen.  If it's a dump screen,
        # there will be no kickers for it, so we just do nothing and return.
        try:
            # We get the kicker setpoints for the screen.
            kicker_setpoints = screen.get_fast_kicker_setpoints()
        except EuXFELUserError: # Then there is no kicker info (e.g. dump screen)
            return
        # Loop over the setpoints corresponding to the screen and write their settings to the machine.
        for setpoint in kicker_setpoints:
            self.kickerop.apply_fast_kicker_setpoint(setpoint)

        # Set the kicker number based on the screen name.  We assume all the setpoints have the same kicker numbers,
        # Otherwise something is very, very wrong in how the DOOCS server is configured (beyond the responsibility of
        # this code).  So we only use the last setpoint from the above loop to set the kicker_number here.
        kmmap = self.sbunches.get_kicker_name_to_kicker_number_map()
        self.sbunches.kicker_number = kmmap[setpoint.name]


class MachineReadManager:
    def __init__(self, *, screens: list[Screen], optics: MachineLinearOptics, request: SnapshotRequest):
        self.screens = screens
        self.optics = optics
        self.snapshotter = Snapshotter(request)

    def full_read(self) -> dict[str, Any]:
        full_optics_read = self.optics.full_read()
        rest_read = self.snapshotter.snapshot(resolve_wildcards=True)
        rest_read |= full_optics_read
        return rest_read
