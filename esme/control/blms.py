from .dint import DOOCSInterface
from .dint import DOOCSAddress

class BeamLossMonitor:
    def __init__(self, name: str, di: DOOCSInterface | None = None) -> None:
        self.name = name
        self.di = di or DOOCSInterface()
        self._fdl = DOOCSAddress.from_string(f"XFEL.DIAG/BLM/{self.name}/")

    def get_slow_counter(self) -> int:
        return self.di.get_value(self._fdl.filled(property="SINGLE_SLOWCOUNTER"))
    
    def get_slow_threshold(self) -> int:
        return self.di.get_value(self._fdl.filled(property="SINGLE_SLOW_PROTECTION_THRESHOLD"))
    
    def slow_protection_reset(self) -> None:
        return self.di.set_value(self._fdl.filled(property="SLOW_PROTECTION.RESET"), 1)
