from enum import Enum
from math import isclose

from .dint import DOOCSInterface


class UncalibratedTDSError(RuntimeError):
    pass


# XFEL.RF/TDS.MODULATOR/TDSA.52.I1.MODULATOR/CHARGE_VOLTAGE.SP


# XFEL.RF/TDS.MODULATOR/TDSB.428.B2/CCPS_UREAD


class StreakingPlane(str, Enum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    UNKNOWN = "UNKNOWN"


class TransverseDeflector:
    PHASE_RB_PROP = "PHASE.SAMPLE"
    AMPLITUDE_RB_PROP = "AMPL.SAMPLE"
    PHASE_SP_PROP = "SP.PHASE"
    AMPLITUDE_SP_PROP = "SP.AMPL"

    def __init__(
        self,
        sp_fdl: str,
        rb_fdl: str,
        modulator_voltage_addr: str,
        fsm_addr: str,
        plane: StreakingPlane,
        calibration=None,
        zero_crossing: float | None = None,
        di: DOOCSInterface | None = None,
    ) -> None:
        """fdl arguments must be of the form /facility/device/location/ with leading and trailing
        forward slashes present."""
        # Setpoint FDL address stub: Facility, Device, Location (only missing Property):
        # E.g.:
        # /XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/
        self.sp_fdl = sp_fdl
        self.rb_fdl = rb_fdl
        self._modulator_voltage_address = modulator_voltage_addr
        self._fsm_address = fsm_addr
        self.plane = plane
        self.calibration = None
        self.zero_crossing = None
        self.di = di if di else DOOCSInterface()

    def get_phase_rb(self) -> float:
        """Get the phase readback value."""
        ch = self.rb_fdl + f"{self.PHASE_RB_PROP}"
        return self.di.get_value(ch)

    def increment_phase(self, phase_increment: float) -> None:
        """Increment the phase by the provided amount."""
        phase_increment += self.get_phase_sp()
        self.set_phase(phase_increment)

    def get_amplitude_rb(self) -> float:
        """Get the amplitude readback value."""
        ch = self.rb_fdl + f"{self.AMPLITUDE_RB_PROP}"
        return self.di.get_value(ch)

    def get_phase_sp(self) -> float:
        """Get the phase setpoint value."""
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        return self.di.get_value(ch)

    def get_amplitude_sp(self) -> float:
        """Get the amplitude setpoint value."""
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        return self.di.get_value(ch)

    def set_phase(self, value: float) -> None:
        """Set the TDS phase"""
        ch = self.sp_fdl + f"{self.PHASE_SP_PROP}"
        self.di.set_value(ch, value)

    def set_amplitude(self, value: float) -> None:
        ch = self.sp_fdl + f"{self.AMPLITUDE_SP_PROP}"
        self.di.set_value(ch, value)

    def set_voltage(self, voltage: float) -> None:
        """Set the TDS voltage by using the member calibration."""
        if self.calibration is None:
            raise UncalibratedTDSError("Missing TDS Calibration")
        self.set_amplitude(self.calibration.get_amplitude(voltage))

    def get_voltage_rb(self) -> float:
        """Get the TDS voltage with the use of the calibration"""
        if self.calibration is None:
            raise UncalibratedTDSError("Missing TDS Calibration")
        return self.calibration.get_voltage(self.get_amplitude_rb())

    def amplitude_rb_matches_sp(self, tol: float = 0.05) -> bool:
        """Check that the amplitude readback value matches the setpoint within some tolerance.
        This is for checking that the TDS is functioning.  Most typically, that it has been
        switched on correctly."""
        rb = self.get_amplitude_sp()
        sp = self.get_amplitude_sp()
        return isclose(rb, sp, abs_tol=tol, rel_tol=tol)

    def set_phase_to_zero_crossing(self) -> None:
        """Set TDS phase to the previously set zero crossing."""
        if self.zero_crossing is None:
            raise ValueError("Zero Crossing has not been set.")
        self.set_phase(self.zero_crossing)

    def set_zero_crossing(self) -> None:
        """Set the zero crossing.  Convenient for jumping back to the zero crossing and not having
        to find it repeatedly."""
        phase = self.get_phase_sp()
        if phase < 0:
            phase %= -180
        else:
            phase %= 180
        self.zero_crossing = phase

    def get_modulator_voltage(self) -> float:
        return self.di.get_value(self._modulator_voltage_addr)

    def get_fsm_state(self) -> int:
        return self.di.get_value(self._fsm_address)
