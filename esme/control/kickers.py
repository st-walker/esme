import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from .dint import DOOCSInterface
from .exceptions import EuXFELUserError


LOG = logging.getLogger(__name__)

class PolarityType(IntEnum):
    POSITIVE = +1
    NEGATIVE = -1


@dataclass
class FastKickerSetpoint:
    name: str
    voltage: float
    delay: int
    polarity: Optional[PolarityType]
    

class KickerOperationError(EuXFELUserError):
    pass


class FastKicker:
    HV_ON_READ_PROP = "DIO_CH.14"
    # What HV_ON_READ_PROP should be if the HV is on.
    HV_ON_READ_PROP_VALUE_ON = 0
    HV_EIN_SET_PROP = "DIO_CH.18"
    HV_AUS_SET_PROP = "DIO_CH.19"
    POSITIVE_SET_PROP = "DIO_CH.21"
    NEGATIVE_SET_PROP = "DIO_CH.22"
    def __init__(self, *, name: str , adio24_fdl:str , trigger_channel:str, di=None):
        self.name = name
        self.adio24_fdl = adio24_fdl
        self.trigger_channel = trigger_channel
        self.di = di if di else DOOCSInterface()
        
    def _full_path(self, leaf):
        return f"{self.adio24_fdl}/{leaf}"

    def set_hv_off(self):
        self.di.set_value(self._full_path(self.HV_AUS_SET_PROP), 1)

    def set_hv_on(self):
        self.di.set_value(self._full_path(self.HV_EIN_SET_PROP), 1)

    def is_hv_on(self):
        val = self.di.get_value(self._full_path(self.HV_ON_READ_PROP))
        return val == self.HV_ON_READ_PROP_VALUE_ON

    def set_polarity(self, polarity: PolarityType) -> None:
        if self.is_hv_on():
            raise KickerOperationError("Trying to change the polarity whilst HV is still on.")
        if polarity is PolarityType.POSITIVE:
            self.di.set_value(f"{self.adio24_fdl}/{self.POSITIVE_SET_PROP}", 1)
        elif polarity is PolarityType.NEGATIVE:
            self.di.set_value(f"{self.adio24_fdl}/{self.NEGATIVE_SET_PROP}", 1)
        else:
            from IPython import embed; embed()
            raise TypeError(f"Unrecognised Polarity {polarity}")

    def set_delay(self, delay: int):
        if delay < 0:
            raise ValueError(f"Negative delay: {delay}")
        channel = f"XFEL.SDIAG/TIMER/{self.trigger_channel}.DELAY"
        self.di.set_value(channel, delay)

    def set_voltage(self, voltage: float):
        if voltage < 0.0:
            raise ValueError("Negative voltage")
        channel = f"XFEL.SDIAG/KICKER.PS/{self.name}/S0"
        self.di.set_value(channel, voltage)

    def get_number(self):
        return self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{self.name}/KICKER_NUMBER")

    def is_operational(self):
        return self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{self.name}/KICKER_STATUS") == 0


class FastKickerController:
    def __init__(self, kickers: list[FastKicker], di=None):
        self.kickers = kickers
        self.di = di if di else DOOCSInterface()

    @property
    def kicker_names(self) -> list[str]:
        return [kicker.name for kicker in self.kickers]

    def get_kicker(self, kicker_name: str) -> FastKicker:
        try:
            kicker_index = self.kicker_names.index(kicker_name)
        except ValueError:
            raise KickerOperationError(f"Unrecognised kicker name {kicker_name}")
        else:
            return self.kickers[kicker_index]

    def apply_fast_kicker_setpoint(self, ksp: FastKickerSetpoint) -> None:
        kicker = self.get_kicker(ksp.name)

        # TODO: this should work when there is no polarity change!
        kicker.set_hv_off()
        if ksp.polarity is not None:
            kicker.set_polarity(ksp.polarity)
        kicker.set_hv_on()
        kicker.set_delay(ksp.delay)
        kicker.set_voltage(ksp.voltage)

