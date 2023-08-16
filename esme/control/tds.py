from .mint import XFELMachineInterface, DOOCSAddress

from .sbunches import DiagnosticRegion

class TransverseDeflector:
    PHASE_RB_PROP = "PHASE.SAMPLE"
    AMPLITUDE_RB_PROP = "AMPL.SAMPLE"
    PHASE_SP_PROP = "SP.PHASE"
    AMPLITUDE_SP_PROP = "SP.AMPL"

    def __init__(self, location: DiagnosticRegion, sp_fdl: str, rb_fdl: str, mi=None) -> None:
        self.location = location
        self.sp_fdl = sp_fdl
        self.rb_fdl = rb_fdl
        self.mi = mi if mi else XFELMachineInterface()

    def get_phase_rb(self) -> float:
        ch = self.rb_fdl + f"{self.PHASE_RB_PROP}"
        return self.mi.get_value(ch)

    def get_amplitude_rb(self) -> float:
        ch = self.rb_fdl + f"{self.AMPLITUDE_RB_PROP}"
        return self.mi.get_value(ch)

    def get_phase_sp(self) -> float:
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        return self.mi.get_value(ch)

    def get_amplitude_sp(self) -> float:
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        return self.mi.get_value(ch)

    def set_phase(self, value: float) -> None:
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        self.mi.set_value(ch, value)

    def set_amplitude(self, value: float) -> None:
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        self.mi.set_value(ch, value)

class TransverseDeflectors:
    def __init__(self, deflectors: list[TransverseDeflector]):
        self.area = DiagnosticRegion("I1")
        self.deflectors = deflectors

    def all_tds_locations(self):
        return [d.location for d in self.deflectors]

    def active_tds(self) -> TransverseDeflector:
        return self.deflectors[self.all_tds_locations().index(self.area)]

    def get_phase_rb(self) -> float:
        return self.active_tds().get_phase_rb()

    def get_amplitude_rb(self) -> float:
        return self.active_tds().get_amplitude_rb()

    def get_phase_sp(self) -> float:
        return self.active_tds().get_phase_sp()

    def get_amplitude_sp(self) -> float:
        return self.active_tds().get_amplitude_sp()

    def set_phase(self, value: float) -> None:
        self.active_tds().set_phase(value)

    def set_amplitude(self, value: float) -> None:
        self.active_tds().set_amplitude(value)
