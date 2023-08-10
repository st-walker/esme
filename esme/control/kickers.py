from .mint import XFELMachineInterface
from enum import Enum, auto
from .exceptions import EuXFELUserError
from dataclasses import dataclass

import logging

LOG = logging.getLogger(__name__)

class PolarityType(Enum):
    POSITIVE = +1
    NEGATIVE = -1

@dataclass
class ScreenConfig:
    name: str
    polarity: PolarityType
    voltage: float
    delay: int
    kicker_name: str
    
class KickerOperationError(EuXFELUserError):
    pass

class Kicker:
    HV_ON_READ = "DIO_CH.14"
    # What HV_ON_READ should be if the HV is on.
    HV_ON_READ_VALUE_ON = 0
    HV_EIN_SET = "DIO_CH.18"
    HV_AUS_SET = "DIO_CH.19"
    POSITIVE_SET = "DIO_CH.21"
    NEGATIVE_SET = "DIO_CH.22"
    def __init__(self, *, name: str , adio24_stem:str , trigger_channel:str, number: int, mi=None):
        self.name = name
        self.adio24_stem = adio24_stem
        self.trigger_channel = trigger_channel
        self.mi = mi if mi else XFELMachineInterface()
        
    def _full_path(self, leaf):
        return f"{self.adio24_stem}/{leaf}"

    def set_hv_off(self):
        self.mi.set_value(self._full_path(self.HV_AUS_SET), 1)

    def set_hv_on(self):
        self.mi.set_value(self._full_path(self.HV_EIN_SET), 1)

    def is_hv_on(self):
        val = self.mi.get_value(self._full_path(self.HV_ON_READ))
        return val == self.HV_ON_READ_VALUE_ON

    def set_polarity(self, polarity: PolarityType) -> None:
        if self.is_hv_on():
            raise KickerOperationError("Trying to change the polarity whilst HV is still on.")
        if polarity is PolarityType.POSITIVE:
            self.mi.set_value(f"{self.adio24_stem}/{self.POSITIVE_SET}", 1)
        elif polarity is PolarityType.NEGATIVE:
            self.mi.set_value(f"{self.adio24_stem}/{self.NEGATIVE_SET}", 1)
        else:
            raise TypeError(f"Unrecognised Polarity {polarity}")

    def set_delay(self, delay: int):
        if delay < 0:
            raise ValueError(f"Negative delay: {delay}")
        channel = f"XFEL.SDIAG/TIMER/{self.trigger_channel}.DELAY"
        self.mi.set_value(channel, delay)

    def set_voltage(self, voltage: float):
        if voltage < 0.0:
            raise ValueError("Negative voltage")
        channel = f"XFEL.SDIAG/KICKER.PS/{self.name}/S0"
        self.mi.set_value(channel, voltage)


class KickerController:
    def __init__(self, screens: list[ScreenConfig], kickers: list[Kicker], mi=None):
        self.screens = screens
        self.kickers = kickers

        self.mi = mi if mi else XFELMachineInterface()

    @property
    def screen_names(self) -> list[str]:
        return [screen.name for screen in self.screens]

    @property
    def kicker_names(self) -> list[str]:
        return [kicker.name for kicker in self.kickers]

    def get_screen_config(self, screen_name: str) -> None:
        try:
            screen_index = self.screen_names.index(screen_name)
        except ValueError:
            raise KickerOperationError(f"Unrecognised screen name {screen_name}")
        else:
            return self.screens[screen_index]

    def get_kicker(self, kicker_name: str) -> None:
        try:
            kicker_index = self.kicker_names.index(kicker_name)
        except ValueError:
            raise KickerOperationError(f"Unrecognised kicker name {kicker_name}")
        else:
            return self.kickers[kicker_index]

    def configure_kicker_for_screen(self, screen_name: str) -> None:
        screen_config = self.get_screen_config(screen_name)
        kicker = self.get_kicker(screen_config.kicker_name)

        kicker.set_hv_off()
        kicker.set_polarity(screen_config.polarity)
        kicker.set_hv_on()
        kicker.set_delay(screen_config.delay)
        kicker.set_voltage(screen_config.voltage)
