from enum import Enum

from .dint import DOOCSInterface

class UncalibratedTDSError(RuntimeError):
    pass

# XFEL.RF/TDS.MODULATOR/TDSA.52.I1.MODULATOR/CHARGE_VOLTAGE.SP


# XFEL.RF/TDS.MODULATOR/TDSB.428.B2/CCPS_UREAD

class StreakingPlane(str, Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"


class TransverseDeflector:
    PHASE_RB_PROP = "PHASE.SAMPLE"
    AMPLITUDE_RB_PROP = "AMPL.SAMPLE"
    PHASE_SP_PROP = "SP.PHASE"
    AMPLITUDE_SP_PROP = "SP.AMPL"

    def __init__(self, sp_fdl: str, rb_fdl: str, plane: StreakingPlane, calibration=None, di=None) -> None:
        self.sp_fdl = sp_fdl
        self.rb_fdl = rb_fdl
        self.plane = plane
        self.calibration = None
        self.di = di if di else DOOCSInterface()

    def get_phase_rb(self) -> float:
        ch = self.rb_fdl + f"{self.PHASE_RB_PROP}"
        return self.di.get_value(ch)

    def get_amplitude_rb(self) -> float:
        ch = self.rb_fdl + f"{self.AMPLITUDE_RB_PROP}"
        return self.di.get_value(ch)

    def get_phase_sp(self) -> float:
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        return self.di.get_value(ch)

    def get_amplitude_sp(self) -> float:
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        return self.di.get_value(ch)

    def set_phase(self, value: float) -> None:
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        self.di.set_value(ch, value)

    def set_amplitude(self, value: float) -> None:
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        self.di.set_value(ch, value)

    def set_voltage(self, voltage: float) -> None:
        try:
            amplitude = self.calibration.get_amplitude(voltage)
        except AttributeError:
            raise UncalibratedTDSError("Missing TDS Calibration")

        self.set_amplitude(amplitude)

    def get_voltage_rb(self) -> float:
        try:
            return self.calibration.get_voltage(self.get_amplitude_rb())
        except AttributeError:
            raise UncalibratedTDSError("Missing TDS Calibration")
        
    def amplitude_rb_matches_sp(self, tol: float = 0.05) -> bool:
        rb = self.get_amplitude_sp()
        sp = self.get_amplitude_sp()
        return abs((rb - sp) / sp) < tol
        
        
