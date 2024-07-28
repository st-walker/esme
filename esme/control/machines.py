from typing import Any

from esme.control.snapshot import SnapshotRequest, Snapshotter
from esme.core import DiagnosticRegion
from esme.calibration import calculate_tds_time_calibration, TimeCalibration

from .dint import DOOCSInterface
from .kickers import FastKickerController
from .optics import MachineLinearOptics
from .sbunches import SpecialBunchesControl
from .scanner import Scanner
from .screens import Screen, Position
from .tds import TransverseDeflector
from .taskomat import Sequence
from .exceptions import DOOCSReadError
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

    def __init__(
        self,
        *,
        scanner: Scanner,
        screen: Screen,
        deflector: TransverseDeflector,
        sbunches: SpecialBunchesControl,
        optics: MachineLinearOptics,
        di=None
    ) -> None:
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
    def __init__(
        self,
        region: DiagnosticRegion,
        kickerop: FastKickerController,
        screens: dict[str, Screen],
        tds: TransverseDeflector,
        optics: MachineLinearOptics,
        sbunches,
        di=None,
    ) -> None:
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

    def __init__(
        self,
        *,
        screens: dict[str, Screen],
        kickerop: FastKickerController,
        sbunches: SpecialBunchesControl,
        di=None
    ):
        di = di or DOOCSInterface()
        self.screens = screens
        self.kickerop = kickerop
        self.sbunches = sbunches

    def set_kickers_for_screen(self, screen_name: str) -> None:
        """For the given screen, set the kicker timings, voltages, etc.
        so that when the kickers are fired, they will kick the diagnostic bunch
        onto the named screen (screen_name)."""
        # from IPython import embed; embed()
        # Get the screen instance by name
        screen = self.screens[screen_name]
        # Try and set the kickers for the screen.  If it's a dump screen,
        # there will be no kickers for it, so we just do nothing and return.
        # We get the kicker setpoints for the screen.
        kicker_setpoints = screen.get_fast_kicker_setpoints()

        # If there are no kickers for this screen then we need to disable kicker activation,
        # Other
        if not kicker_setpoints:
            self.sbunches.dont_use_kickers()
            return

        # Loop over the setpoints corresponding to the screen and write their settings to the machine.
        for setpoint in kicker_setpoints:
            self.kickerop.apply_fast_kicker_setpoint(setpoint)

        print(
            screen_name,
            setpoint.name,
            "???????????????????????????????????!!!!!!!!!!!!!!!!!!!",
        )
        # Set the kicker number by name using one of the setpoints.
        self.sbunches.set_kicker_name(setpoint.name)


class MachineReadManager:
    def __init__(
        self,
        *,
        screens: dict[str, Screen],
        optics: MachineLinearOptics,
        request: SnapshotRequest,
    ):
        self.screens = screens
        self.optics = optics
        self.snapshotter = Snapshotter(request)

    def full_read(self) -> dict[str, Any]:
        full_optics_read = self.optics.full_read()
        rest_read = self.snapshotter.snapshot(resolve_wildcards=True)
        rest_read |= full_optics_read
        return rest_read


class ImagingManager(MachineReadManager):
    def __init__(
        self,
        *,
        screens: dict[str, Screen],
        optics: MachineLinearOptics,
        request: SnapshotRequest,
        deflector: TransverseDeflector,
        sbunches: SpecialBunchesControl,
        time_calibrations: dict[str, TimeCalibration] | None = None
    ):
        super().__init__(screens=screens, optics=optics, request=request)
        self.deflector = deflector
        self.sbunches = sbunches
        self.time_calibrations = time_calibrations or {}
        self.di = DOOCSInterface()

    def calculate_time_calibration_from_voltage(self, screen_name: str) -> float:
        r12 = self.optics.r12_streaking_from_tds_to_point(screen_name)            
        voltage = self.deflector.get_voltage_rb()
        beam_energy = self.optics.get_beam_energy()
        print(f"{r12=}, {voltage=}, {beam_energy=}")
        return calculate_tds_time_calibration(r12_streaking=r12,
                                              energy_mev=beam_energy,
                                              voltage=voltage)
    
    def calculate_time_calibration_from_calibration(self, screen_name: str) -> float:
        amplitude = self.deflector.get_amplitude_rb()
        return self.time_calibrations[screen_name].get_time_calibration(amplitude)

    def is_tds_calibrated(self) -> bool:
        return self.tds.calibration is not None
    
    def is_screen_time_calibrated(self, screen_name: str) -> bool:
        return screen_name in self.time_calibrations

    def turn_beam_off(self) -> None:    
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def turn_beam_on(self) -> None:
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def is_beam_on(self) -> bool:
        return bool(self.di.get_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"))

    def is_beam_on_screen(self, screen: Screen) -> bool:
        pos = screen.get_position()
        is_beam_off_axis = self.sbunches.is_diag_bunch_firing() and pos is Position.OFFAXIS
        is_beam_on_axis = self.is_beam_on() and pos is Position.ONAXIS
        return is_beam_on_axis or is_beam_off_axis

    def take_beam_off_screen(self, screen: Screen) -> None:
        position = screen.get_position()
        if position is Position.ONAXIS:
            self.sbunches.stop_diagnostic_bunch()
            self.turn_beam_off()
            # We don't clip off-axis artefacts if the beam is on axis
            screen.analysis.set_clipping(on=False)
        elif position is Position.OFFAXIS:
            self.sbunches.stop_diagnostic_bunch()
            # We are off axis so we enable clipping of off axis rubbish
        elif position is Position.OUT:
            raise RuntimeError(f"Screen {screen.name} is out")

        # Tidy up by disabling ROI clipping in the image analysis server
        # Maybe not really that important but just in case/nice to do.
        screen.analysis.set_clipping(on=False)

    def turn_beam_onto_screen(self, screen: Screen) -> None:
        position = screen.get_position()
        if position is Position.ONAXIS:
            self.turn_beam_on()
            # We clip the off-axis artefacts if the beam is on axis
            screen.analysis.set_clipping(on=False)
            self.sbunches.dont_use_kickers()
            self.sbunches.start_diagnostic_bunch()
        elif position is Position.OFFAXIS:
            self.turn_beam_on()
            self.sbunches.start_diagnostic_bunch()
            # We are off axis so we enable clipping of off axis rubbish
            screen.analysis.set_clipping(on=True)
        elif position is Position.OUT:
            raise RuntimeError(f"Screen {screen.name} is out")


class DumpManager:
    def __init__(self, *, forward_sequence: Sequence, backward_sequence: Sequence):
        self.forward_sequence = forward_sequence
        self.backward_sequence = backward_sequence

class TDSCalibrationManager:
    def __init__(self,
                kickerop: FastKickerController,
                screens: dict[str, Screen],
                tds: TransverseDeflector,
                optics: MachineLinearOptics,
                sbunches: SpecialBunchesControl,
                di: DOOCSInterface | None = None):
        self.kickerop = kickerop
        self.screens = screens
        self.tds = tds
        self.optics = optics
        self.sbunches = sbunches
        self.di = di or DOOCSInterface()

    def set_kickers_for_screen(self, screen_name: str):
        _set_kickers_for_screen(self.kickerop, self.sbunches, self.screens[screen_name])

    def turn_beam_on(self) -> None:
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def turn_beam_off(self) -> None:
        self.di.set_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def is_beam_allowed(self) -> bool:
        return bool(self.di.get_value("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"))

    def turn_beam_onto_screen(self, screen: Screen, streak: bool = True) -> None:
        position = screen.get_position()
        self.sbunches.set_use_tds(use_tds=streak)
        if position is Position.ONAXIS:
            self.turn_beam_on()
            # We clip the off-axis artefacts if the beam is on axis
            screen.analysis.set_clipping(on=False)
            self.sbunches.set_to_last_bunch()
            self.sbunches.dont_use_kickers()
            self.sbunches.start_diagnostic_bunch()
        elif position is Position.OFFAXIS:
            self.turn_beam_on()
            self.kick_beam_onto_screen(screen)
            # We are off axis so we enable clipping of off axis rubbish
            screen.analysis.set_clipping(on=True)
        elif position is Position.OUT:
            raise RuntimeError(f"Screen {screen.name} is out")

    def take_beam_off_screen(self, screen: Screen) -> None:
        position = screen.get_position()
        if position is Position.ONAXIS:
            self.sbunches.stop_diagnostic_bunch()
            self.turn_beam_off()
            # We don't clip off-axis artefacts if the beam is on axis
            screen.analysis.set_clipping(on=False)
        elif position is Position.OFFAXIS:
            self.sbunches.stop_diagnostic_bunch()
            # We are off axis so we enable clipping of off axis rubbish
        elif position is Position.OUT:
            raise RuntimeError(f"Screen {screen.name} is out")

        # Tidy up by disabling ROI clipping in the image analysis server
        # Maybe not really that important but just in case/nice to do.
        screen.analysis.set_clipping(on=False)

    def kick_beam_onto_screen(self, screen: Screen) -> None:
        # Set the kickers for the screen
        self.set_kickers_for_screen(screen.name)
        # Append a diagnostic bunch by setting bunch to last in machine + 1
        self.sbunches.set_to_append_diagnostic_bunch()
        # Now start firing the fast kicker(s).
        self.sbunches.start_diagnostic_bunch()


def _set_kickers_for_screen(kickerop: FastKickerController,
                            sbunches: SpecialBunchesControl, 
                            screen: Screen) -> None:
    """For the given screen, set the kicker timings, voltages, etc.
    so that when the kickers are fired, they will kick the diagnostic bunch
    onto the named screen (screen_name)."""
    # Get the screen instance by name
    # Try and set the kickers for the screen.  If it's a dump screen,
    # there will be no kickers for it, so we just do nothing and return.
    # We get the kicker setpoints for the screen.
    kicker_setpoints = screen.get_fast_kicker_setpoints()

    # If there are no kickers for this screen then we need to disable kicker activation,
    # Other
    if not kicker_setpoints:
        sbunches.dont_use_kickers()
        return

    # Loop over the setpoints corresponding to the screen and write their settings to the machine.
    for setpoint in kicker_setpoints:
        kickerop.apply_fast_kicker_setpoint(setpoint)

    # Set the kicker number by name using one of the setpoints.
    sbunches.set_kicker_name(setpoint.name)

