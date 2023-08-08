"""Module for calibrating the TDS.  You tell it: I want these voltages
and it goes away and does it for you..."""
from dataclasses import dataclass

from .dispersion import QuadrupoleSetting

from .mint import Machine

import logging

LOG = logging.getLogger(__name__)


@dataclass
class TDSScanConfiguration:
    voltages: list
    quad_setting: QuadrupoleSetting
    # scan_dispersion: float


@dataclass
class SetpointReadbackPair:
    setpoint: str
    readback: str

@dataclass
class TDSAddresses:
    amplitude: SetpointReadbackPair
    phase: SetpointReadbackPair
    event: str
    bunch_one: str


class MockTDS:
    pass

class TDSController:
    RB_SP_TOLERANCE = 0.02
    def __init__(self, addresses: TDSAddresses, mi=None):
        if mi is None:
            mi = XFELMachineInterface()
        self._mi = mi
        self.addies = addresses
        self.bunch_one_timing = self.read_on_beam_timing()

    def set_amplitude(self, amplitude: float) -> None:
        """Set the TDS amplitude"""
        LOG.debug(f"Setting TDS amplitude: {self.addies.amplitude.setpoint} @ {amplitude}")
        self._mi.set_value(self.addies.amplitude.setpoint, amplitude)

    def read_rb_amplitude(self) -> float:
        """Read back the TDS amplitude"""
        result = self._mi.get_value(self.addies.amplitude.readback)
        LOG.debug(f"Reading TDS amplitude: {self.addies.amplitude.readback} @ {result}")
        return result

    def read_sp_amplitude(self) -> float:
        """Read back the TDS amplitude"""
        result = self._mi.get_value(self.addies.amplitude.setpoint)
        LOG.debug(f"Reading TDS amplitude: {self.addies.amplitude.readback} @ {result}")
        return result

    def set_phase(self, phase: float) -> None:
        LOG.debug(f"Setting TDS amplitude: {self.addies.phase.setpoint} @ {phase}")
        self._mi.set_value(self.addies.phase.setpoint, phase)

    def read_rb_phase(self) -> float:
        result = self._mi.get_value(self.addies.phase.readback)
        LOG.debug(f"Reading TDS amplitude: {self.addies.phase.readback} @ {result}")
        return result

    def read_on_beam_timing(self):
        return self._mi.get_value(self.addies.bunch_one)

    def is_powered(self) -> bool:
        LOG.debug("Checking if TDS is powered")
        rb = self.read_rb_amplitude()
        sp = self.read_sp_amplitude()
        relative_difference = abs(rb - sp) / sp
        powered = relative_difference < self.RB_SP_TOLERANCE
        LOG.debug(f"TDS RB ampl = {rb}; TDS SP = {sp}: {relative_difference=} -> {powered}")
        return powered

    def read_timing(self):
        return self._mi.get_value(self.addies.event)[2]

    def read_on_beam_timing(self) -> list:
        return self._mi.get_value(self.addies.bunch_one)

    def is_on_beam(self) -> bool:
        LOG.debug("Checking if TDS is on beam")
        return self.read_timing() == self.bunch_one_timing

    def switch_off_beam(self) -> None:
        """Turn the TDS off beam (whilst keeping RF power)"""
        LOG.debug(f"Setting TDS off beam")
        self._switch_tds_on_off_beam(on=False)

    def switch_on_beam(self) -> None:
        """Turn the TDS on beam"""
        LOG.debug(f"Setting TDS on beam")
        self._switch_tds_on_off_beam(on=True)

    def _switch_tds_on_off_beam(self, *, on: bool) -> None:
        bunch_number = 1 # Hardcoded, always nuch
        on_data = [bunch_number, int(on), 0, 0]  # 3rd: "kicker", 4th: "WS-subtrain"

        on_data = [bunch_number, 111, self.read_on_beam_timing(), 1]
        if not on:
            on_data[2] *= 10000  # Simply a big number and so very far from being on beam.
        self._mi.set_value(self.addies.event, on_data)


class TDSCalibratingMachine(Machine):
    def __init__(self, outdir, *, screen_channel, template):
        super().__init__(self.make_template(outdir))
        self.tds = self.TDSCLS()
        self.mi = XFELMachineInterface()
        self.nphase_points = 60

    def calibrate(self, amplitudes, dirout=None):
        for amplitude in amplitudes:
            self.tds.set_amplitude(amplitude)
            slope = self.get_slope(amplitude)

    def scan_phase(self):
        npoints = len(self.phase)

        for i, _ in enumerate(self.phases):
            self.tds.set_phase(phase)
            screen = self.get_screen_image()

    def get_slope(self, amplitude):
        phases = np.linspace(-200, 200, self.nphase_points)
        amplitude = self.tds.read_sp_amplitude()
        outdir = pathlib.Path("tds-calibration")
        outdir = (outdir / f"amplitude={int(amplitude)}")
        outdir.mkdir(exist_ok=True, parents=True)
        import pickle
        ycoms = []
        total = []
        all_images = []
        import pickle
        print("Phase is currently", self.tds.read_rb_phase())
        for phase in phases:
            print("setting to phase: ", phase)
            self.tds.set_phase(phase)
            screen = self.get_screen_image()
            time.sleep(1.1)
            from esme.image import process_image
            all_images = process_image(screen, 0)
            from scipy.ndimage import center_of_mass
            ycoms.append(center_of_mass(screen)[1])
            total.append(screen.sum())
            outpath =  (outdir / str(phase)).with_suffix(".npz")
            with outpath.open("wb") as f:
                np.savez(outpath, screen)

    def get_screen_image(self):
        """Get screen image"""
        channel = self.SCREEN_CHANNEL
        LOG.debug(f"Reading image from {channel}")
        return self.mi.get_value(channel)
