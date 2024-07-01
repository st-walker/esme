import re

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
        self.di.set_value(self._fdl.resolve(property="RUN.ONCE"), 1)

    def is_running(self) -> bool:
        return self.di.get_value(self._fdl.resolve(property="RUNNING")) == 1

    def _get_step_address(self, step_number: int, suffix: str) -> str:
        # given step number and suffix, gives full address.
        return self._fdl.resolve(property=f"STEP{step_number:03}.{suffix}")

    @cache
    def _get_step_numbers(self) -> list[int]:
        properties = self.di.names(str(self._fdl.resolve(property="*")))
        matches = [_STEP_REGEX.match(m) for p in properties]
        matches = [m for m in matches if m]
        step_numbers = set([m.groups(1) for m in matches])
        return sorted(step_numbers)

    def get_running_step(self) -> int:
        for step_number in self._get_n_steps():
            if self.is_step_running(step_number):
                return step_number
        raise SequenceNotRunningError(f"Sequence: {self} is not running")

    def get_number_of_steps(self) -> int:
        return len(self._get_n_steps())

    def is_step_running(self, step_number: int) -> bool:
        return self._get_step_address(step_number, "RUNNING") == 1

    def is_step_disabled(self, step_number: int) -> bool:
        return self._get_step_address(step_number, "DISABLED") == 1

    def is_step_error(self, step_numnber: int) -> bool:
        return self._get_step_address(step_number, "STATE") == 2

    def get_step_label(self, step_number: int) -> str:
        return self._get_step_address(step_number, "LABEL")

    def get_label(self) -> str:
        return self.di.get_value(self._fdl.resolve(property="LABEL"))

    def get_html_log(self) -> str:
        self.di.get_value(self._fdl.resolve(property="LOG_HTML"))
