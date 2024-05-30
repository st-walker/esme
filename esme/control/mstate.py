from dataclasses import dataclass
from enum import Enum, auto

from .dint import DOOCSInterface
from .exceptions import DOOCSReadError
from .kickers import FastKickerController
from .sbunches import SpecialBunchesControl
from .screens import Screen
from .tds import TransverseDeflector

LH_CLOSED_ADDRESS = "XFEL.UTIL/LASERINT/GUN/SH3_CLOSED"
LH_OPEN_ADDRESS = "XFEL.UTIL/LASERINT/GUN/SH3_OPEN"


class Health(Enum):
    GOOD = auto()
    WARNING = auto()
    BAD = auto()
    UNKNOWN = auto()
    SUBJECTIVE = auto()


@dataclass
class Condition:
    health: Health
    short: str = ""
    long: str = ""


class AreaWatcher:
    def __init__(
        self,
        screens: dict[str, Screen],
        kickerop: FastKickerController,
        tds: TransverseDeflector,
        sbunches: SpecialBunchesControl,
        watched_screen_name: str | None = None,
        di: DOOCSInterface | None = None,
    ):
        self._screens = screens
        self._kickerop = kickerop
        self._tds = tds
        self._sbunches = sbunches
        self.watched_screen_name = next(iter(self._screens))
        self.di = di if di else DOOCSInterface()

    def get_laser_heater_shutter_state(self) -> Condition:
        try:
            is_open = self.di.get_value(LH_OPEN_ADDRESS)
            is_closed = self.di.get_value(LH_CLOSED_ADDRESS)
        except DOOCSReadError as e:
            return Condition(
                Health.BAD,
                short="DOOCS ERROR",
                long=f"Unable to read {e.address} when checking IBFB",
            )

        if is_open:
            return Condition(Health.SUBJECTIVE, short="OPEN", long="LH shutter is open")

        if is_closed:
            return Condition(
                Health.SUBJECTIVE, short="CLOSED", long="LH shutter is closed"
            )

        return Condition(
            Health.BAD, short="ERROR", long="Unable to determine LH shutter state"
        )

    def _get_watched_screen(self) -> Screen:
        return self._screens[self.watched_screen_name]

    def check_screen_state(self) -> Condition:
        tooltips = []
        try:
            screen = self._get_watched_screen()
        except KeyError:
            return Condition(Health.UNKNOWN, long="No Screen Name Set")
        try:
            if not screen.is_powered():
                tooltips.append("• Camera is off")
            if screen.is_offaxis():
                tooltips.append("• Camera is out")
            if screen.is_acquiring_images():
                tooltips.append("• Camera is not taking data")
            if not self._sbunches.is_screen_ok():
                tooltips.append("• SBM is unhappy with the screen")
            if (
                tooltips
            ):  # If something's wrong then read camera status to try to help perhaps.
                tooltips.append(f'• Camera status: "{screen.read_camera_status()}"')
        except DOOCSReadError as e:
            tooltips.append("Unexpected read error whilst checking\nscreen state.")

        tooltip = "\n".join(tooltips)
        state = Health.GOOD if not bool(tooltips) else Health.BAD
        return Condition(state, long=tooltip)

    def check_tds_state(self) -> Condition:
        tooltips = []
        try:
            if not self._sbunches.is_tds_ok():
                tooltips.append("• SBM is unhappy with the TDS")
            if not self._tds.amplitude_rb_matches_sp():
                rb = self._tds.get_amplitude_rb()
                sp = self._tds.get_amplitude_sp()
                tooltips.append(
                    f"• TDS amp. failing to reach setpoint: SP={sp}, RB={rb}"
                )
        except DOOCSReadError as e:
            tooltips.append("Unexpected read error whilst checking\nTDS state.")
        tooltip = "\n".join(tooltips)
        state = Health.GOOD if not bool(tooltips) else Health.BAD
        return Condition(state, long=tooltip)

    def check_kickers_state(self) -> Condition:
        tooltips = []
        try:
            screen = self._get_watched_screen()
        except KeyError:
            return Condition(
                Health.UNKNOWN, long="No screen name set, unable to determine kicker"
            )

        kicker_setpoints = screen.get_fast_kicker_setpoints()
        try:
            for kicker_setpoint in kicker_setpoints:
                name = kicker_setpoint.name
                kicker = self._kickerop.get_kicker(name)
                if not kicker.is_operational() or not self._sbunches.is_kicker_ok():
                    tooltips.append(f"• SBM is complaining about {name}")
                if not kicker.is_hv_on():
                    tooltips.append(f"• HV is not on for kicker {name}")
        except DOOCSReadError as e:
            tooltips.append(
                f"Unexpected read: {e.address} error whilst checking\nkicker state"
            )
        tooltip = "\n".join(tooltips)
        state = Health.GOOD if not bool(tooltips) else Health.BAD
        return Condition(state, long=tooltip)

    def get_ibfb_state(self) -> Condition:
        # Assume ibfb is off
        ibfb = False
        try:
            xon = self._sbunches.ibfb_x_lff_is_on()
            yon = self._sbunches.ibfb_y_lff_is_on()
        except DOOCSReadError as e:
            return Condition(
                Health.SUBJECTIVE,
                short="UNKNOWN",
                long=f"Unable to read {e.address} when checking IBFB",
            )

        if not (xon or yon):
            return Condition(
                Health.SUBJECTIVE,
                short="OFF",
                long="IBFB AFF is off, as it should be when operating the SBM",
            )

        tooltips = []
        if xon:
            tooltips.append("• IBFB X is on")

        if yon:
            tooltips.append("• IBFB Y is on")

        return Condition(Health.SUBJECTIVE, short="ON", long="\n".join(tooltips))
