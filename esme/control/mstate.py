from typing import Optional

from .dint import DOOCSInterface
from .screens import Screen
from .sbunches import SpecialBunchesControl
from .exceptions import DOOCSReadError
from .kickers import FastKickerController
from .tds import TransverseDeflector

LH_CLOSED_ADDRESS = "XFEL.UTIL/LASERINT/GUN/SH3_CLOSED"
LH_OPEN_ADDRESS = "XFEL.UTIL/LASERINT/GUN/SH3_OPEN"

class AreaWatcher:
    def __init__(self, screens: Optional[dict[str, dict[str, Screen]]] = None,
                 kickerop: Optional[FastKickerController] = None,
                 tds: Optional[TransverseDeflector] = None,
                 sbunches: Optional[SpecialBunchesControl] = None,                 
                watched_screen_name: Optional[str] = None,
                di : Optional[DOOCSInterface] = None):
        self._screens = screens
        self._kickerop = kickerop
        self._tds = tds
        self._sbunches = sbunches
        self.watched_screen_name = None
        self.di = di if di else DOOCSInterface()

    def is_laser_heater_shutter_open(self) -> bool:
        is_open = self.di.get_value(LH_OPEN_ADDRESS)
        is_closed = self.di.get_value(LH_CLOSED_ADDRESS)
        return is_open and not is_closed
    
    def _get_watched_screen(self) -> Screen:
        return self._screens[self.watched_screen_name]
    
    def check_screen_state(self) -> tuple[bool, str]:
        tooltips = []
        try:
            screen = self._get_watched_screen()
        except:
            return False, "No Screen Name Set"
        try:
            if not screen.is_online():
                tooltips.append("• Camera is off")
            if screen.is_offaxis():
                tooltips.append("• Camera is out")
            if screen.is_camera_taking_data():
                tooltips.append("• Camera is not taking data")
            if not self._sbunches.is_screen_ok():
                tooltips.append("• SBML is unhappy with the screen")
            if tooltips: # If something's wrong then read camera status to try to help.
                tooltips.append(f"• Camera status: \"{screen.read_camera_status()}\"")
        except DOOCSReadError as e:
            tooltips.append("Unexpected read error whilst checking\nscreen state.")
    
        tooltip = "\n".join(tooltips)
        return not bool(tooltips), tooltip

    def check_tds_state(self) -> tuple[bool, str]:
        tooltips = []
        try:
            if not self._sbunches.is_tds_ok():
                tooltips.append("• SBML is unhappy with the TDS")
            if not self._tds.amplitude_rb_matches_sp():
                rb = self._tds.get_amplitude_rb()
                sp = self._tds.get_amplitude_sp()
                tooltips.append(f"• TDS amp. failing to reach setpoint: SP={sp}, RB={rb}")
        except DOOCSReadError as e:
            tooltips.append("Unexpected read error whilst checking\nTDS state.")
        tooltip = "\n".join(tooltips)
        return not bool(tooltips), tooltip
    
    def check_kickers_state(self) -> tuple[bool, str]:
        tooltips = []
        try:
            screen = self._get_watched_screen()
        except KeyError:
            return False, "No screen name set, unable to determine kicker"

        kicker_setpoints = screen.get_fast_kicker_setpoints()
        try:
            for kicker_setpoint in kicker_setpoints:
                name = kicker_setpoint.name
                kicker = self._kickerop.get_kicker(name)
                if not kicker.is_operational() or not self.is_kicker_ok():
                    tooltips.append(f"• SBML is complaining about {name}")
                if not kicker.is_hv_on():
                    tooltips.append(f"• HV is not on for kicker {name}")
        except DOOCSReadError as e:
            tooltips.append("Unexpected read error whilst checking\nkicker state")
        tooltip = "\n".join(tooltips)
        return not bool(tooltips), tooltip
            

        