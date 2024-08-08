import re
from functools import cache
from typing import Any

from .dint import DOOCSAddress, DOOCSInterface

_STEP_REGEX = re.compile(
    r"""STEP           # STEP
    (?P<step>          # bind the 3 digits to group "step"
    \d{3}              # 3 digits
    )
    \.
    [A-Z0-9_.]+        # 1 or more caps, numbers or underscores
    """,
    re.VERBOSE,
)


class SequenceNotRunningError(Exception):
    pass


class Sequence:
    def __init__(self, location: str, di: DOOCSInterface | None = None):
        self._location = location
        self._fdl = DOOCSAddress.from_string(f"XFEL.UTIL/TASKOMAT/{location}/")
        self.di = di or DOOCSInterface()

    @property
    def location(self) -> str:
        return self._location

    def run_once(self) -> None:
        self.di.set_value(self._fdl.filled(property="RUN.ONCE"), 1)

    def is_running(self) -> bool:
        return self.di.get_value(self._fdl.filled(property="RUNNING")) == 1

    def force_stop(self) -> None:
        return self.di.set_value(self._fdl.filled(property="FORCESTOP"), 1)

    def run_step(self, step_number: int) -> None:
        self.di.set_value(self._get_step_address(step_number, "RUN"), 1)

    def _get_step_address(self, step_number: int, suffix: str) -> str:
        # given step number and suffix, gives full address.
        return self._fdl.filled(property=f"STEP{step_number:03}.{suffix}")

    @cache
    def _get_step_type_map(self) -> dict[int, str]:
        addy = self._fdl.filled(property="COMBOBOX_TYPES")
        combobox_types = self.di.get_value(addy)
        result = {}
        for entry in combobox_types.split(";"):
            step_type, type_number = entry.split("|")
            result[int(type_number)] = step_type

        return result

    def get_step_type(self, step_number: int) -> str:
        step_type_int = self._get_step_value(step_number, "TYPE")
        return self._get_step_type_map()[step_type_int]

    @cache
    def get_step_numbers(self) -> list[int]:
        properties = self.di.get_names(str(self._fdl.filled(property="*")))
        matches = [_STEP_REGEX.match(p) for p in properties]
        matches = [m for m in matches if m]
        steps = set([int(m.group("step")) for m in matches])
        return sorted(steps)

    def get_running_step(self) -> int:
        for step_number in self.get_step_numbers():
            if self.is_step_running(step_number):
                return step_number
        raise SequenceNotRunningError(f"Sequence: {self} is not running")

    def _get_step_value(self, step_number: int, suffix: str) -> Any:
        address = self._get_step_address(step_number, suffix)
        return self.di.get_value(address)

    def get_number_of_steps(self) -> int:
        return len(self.get_step_numbers())

    def is_step_running(self, step_number: int) -> bool:
        return self._get_step_value(step_number, "RUNNING") == 1

    def is_step_disabled(self, step_number: int) -> bool:
        return self._get_step_value(step_number, "DISABLED") == 1

    def is_step_error(self, step_number: int) -> bool:
        return self._get_step_value(step_number, "STATE") == 2

    def get_step_label(self, step_number: int) -> str:
        return self._get_step_value(step_number, "LABEL")

    def get_label(self) -> str:
        return self.di.get_value(self._fdl.filled(property="LABEL"))

    def get_html_log(self) -> str:
        return self.di.get_value(self._fdl.filled(property="LOG_HTML"))

    def set_dynamic_property(self, prop: str, value: Any) -> None:
        self.di.set_value(self._fdl.filled(property=prop), value)
