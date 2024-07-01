from dataclasses import dataclass


@dataclass
class CalibrationContext:
    amplitude: float
    screen_name: str  # Drop down menu?  Automatically calculated from
    # machine when generating new row or whatever.  Unless loading?
    # then we don't want to look it up.  whenever we update snapshot,
    # we update this.  the question then is only if/when we update the
    # snapshot.
    snapshot: pd.Series
    beam_energy: float
    frequency: float
    # button to add new row?

    # stacked widgets: one table for i1, one for b2.
    # No need to show screen, user can just see it elsewhere.

    # Add new row = instantiates above with current values from machine.
    # when new screen is selected, r12_streaking updates.
    # initial screen name is just same as previous row's.S
    # but with phases blank.

    def r12_streaking(self) -> float:
        pass


@dataclass
class PhaseScan:
    """Phase scan. This is output from the calibration routine."""

    prange0: tuple[float, float]
    samples: int
    coms: npt.NDArray | None


@dataclass
class CalibrationSetpoint:
    """This is the combination of the input with its corresponding output."""

    context: CalibrationContext
    pscan0: PhaseScan
    pscan1: PhaseScan

    def voltages(self):
        pass
