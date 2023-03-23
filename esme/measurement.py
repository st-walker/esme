# Priority:

# Should keep track of good images/bad images.  Ignore bad ones.  Should already be possible?  using TDS ampl rb, etc...
# Should check TDS is on, check beam is on, check screen is out.


# Checklist:

# Do the dispersion measurement
# Image processing.
# Make sure it is fully mocked and tested

# Handle the case when the TDS was off.  and don't increment the counter!

# Automatically calibrate the TDS.

# Misc:
# Write metadata: measured dispersion, measured beta_x (if we do that): DONE
# Detect when there is no TDS RF power.
# Detect when the TDS is on or off beam.
# Handle the TDS turning on/off randomly.
# Handle the beam turning off randomly.
# Check screen is out.
# Autogain for images.
# Do betascan too eventually (actually isn't that hard...)
# put tscan, dscan or bscan into the filenames
# if the readback and setpoints differ by a lot then it should complain/crash loudly. including quads!
# Ultimately the files should be saved in some "standard place"
# MeasurementRunner "name" should/could be automatically generated for the eventual gui.
# the toml config may ultimately need to be a class when configured using the GUI...
# Also assert there's only one beam in the machine please.
# MAke a checklist for me to test these features/checks!

# Questions for Bolko:

# Priority:
# Dispersion and TDS scans robustly, without measuring the dispersion
# Measure the dispersion at each quadrupole setting
# Calibrate the TDS

# Possible improvements:
# Do away with raw pickled DFs and instead pickle something more friendly that HAS a df.

from __future__ import annotations

from dataclasses import dataclass
import logging
import time
import os
from pathlib import Path
import pickle
from typing import TypeVar, Optional, Type, Union
from enum import Enum, auto


import toml
import pandas as pd
from textwrap import dedent
import numpy as np

from esme.mint import MPS, Machine, XFELMachineInterface
from esme.channels import (make_injector_snapshot_template,
                           make_b2d_snapshot_template,
                           I1D_SCREEN_ADDRESS,
                           TDS_I1_AMPLITUDE_READBACK_ADDRESS,
                           BUNCH_ONE_TDS_I1,
                           BUNCH_ONE_TDS_B2,
                           EVENT10_CHANNEL,
                           EVENT12_CHANNEL,
                           BUNCH_ONE_TDS_B2,
                           B2D_SCREEN_ADDRESS,
                           BUNCH_ONE_TOLERANCE)

TDSScanConfigurationSelf = TypeVar("TDSScanConfigurationSelfType", bound="TDSScanConfiguration")
DispersionScanConfigurationSelf = TypeVar("DispersionScanConfigurationSelfType", bound="DispersionScanConfiguration")


LOG = logging.getLogger(__name__)

DELAY_AFTER_BEAM_OFF: float = 5.0


PIXEL_SCALE_X_M: float = 13.7369e-6
PIXEL_SCALE_Y_M: float = 11.1756e-6

# what im doing now is writing some docstrings and then also making it
# so that the output, when combined with the scan.toml, is easy to
# resume from.


class EnergySpreadMeasurementError(RuntimeError):
    pass


@dataclass
class QuadrupoleSetting:
    """Class for storing a single strength configuration of the
    quadrupole scan, as well as the intended dispersion it is supposed
    to achieve at the screen.

    The integrated strengths and are assumed throughout this package to have units of m^{-1}.

    """

    names: list
    k1ls: list
    dispersion: float

    def __post_init__(self):  # TODO: test this!
        if len(self.names) != len(self.k1ls):
            raise ValueError("Length mismatch between names and integrated strengths")

    def k1l_from_name(self, name):
        index = self.names.index(name)
        return self.k1ls[index]


@dataclass
class DispersionScanConfiguration:
    """A series of QuadrupoleSetting instances for each datapoint in
    the dispersion scan, as well as the "reference_setting"---used in
    the TDS scan and set at the start of the dispersion scan."""

    reference_setting: QuadrupoleSetting
    scan_settings: list[QuadrupoleSetting]

    @property
    def dispersions(self) -> list[float]:
        return [qsetting.dispersion for qsetting in self.scan_settings]

@dataclass
class TDSScanConfiguration:
    reference_amplitude: float
    scan_amplitudes: list
    scan_dispersion: float


class MeasurementRunner:
    SLEEP_BETWEEN_SNAPSHOTS = 1
    SLEEP_AFTER_TDS_SETTING = 0.5
    SLEEP_AFTER_QUAD_SETTING = 0.5

    def __init__(
            self,
            outdir: Union[os.PathLike, str],
            dscan_config: DispersionScanConfiguration,
            tds_config: TDSScanConfiguration,
            machine: Optional[EnergySpreadMeasuringMachine],
            mps: Optional[MPS] = None,
            dispersion_measurer: Optional[Type[BaseDispersionMeasurer]] = None,
    ):
        """name is used for the output file name"""
        self.outdir = Path(outdir)
        self.dscan_config = dscan_config
        self.tds_config = tds_config

        self.machine = machine
        if self.machine is None:
            templ = make_injector_snapshot_template(self.abs_output_directory())
            self.machine = EnergySpreadMeasuringMachine(templ)

        self.photographer = ScreenPhotographer(machine=self.machine, mps=mps)
        self.dispersion_measurer = dispersion_measurer
        if self.dispersion_measurer is None:
            LOG.debug("Using BasicDispersionMeasurer: will not automatically measure dispersion.")
            self.dispersion_measurer = BasicDispersionMeasurer()

    def run(self, *, bg_shots: int, beam_shots: int) -> None:
        self.tds_scan(bg_shots=bg_shots, beam_shots=beam_shots)
        self.dispersion_scan(bg_shots=bg_shots, beam_shots=beam_shots)

    def tds_scan(self, *, bg_shots: int, beam_shots: int) -> list[Path]:
        """Do the TDS scan for this energy spread measurement"""
        LOG.info("Setting up TDS scan")
        dispersion, dispersion_unc = self.reset_and_measure_dispersion()
        dsetpoint = self.dscan_config.reference_setting.dispersion

        tds_amplitudes = self.tds_config.scan_amplitudes
        LOG.info(f"starting TDS scan: {tds_amplitudes=}, {bg_shots=}, {beam_shots=}")
        self.machine.tds.switch_on_beam()
        filenames = []
        for ampl in tds_amplitudes:
            setpoint_snapshots = self.tds_scan_one_measurement(
                tds_amplitude=ampl,
                dispersion_setpoint=dsetpoint,
                measured_dispersion=(dispersion, dispersion_unc),
                bg_shots=bg_shots,
                beam_shots=beam_shots,
            )

            fname = self.save_setpoint_snapshots(setpoint_snapshots)
            filenames.append(fname)

        LOG.info(f"Finished TDS scan: output pcl: {filenames=}")
        return filenames

    def tds_scan_one_measurement(
        self,
        *,
        tds_amplitude: float,
        dispersion_setpoint: float,
        measured_dispersion: tuple[float, float],
        bg_shots: int,
        beam_shots: int,
    ) -> SetpointSnapshots:
        """Do a single tds scan measurement at the given TDS
        amplitude, with a variable number of background images
        (bg_shots), beam images (beam_shots) and a delay between each
        image being taken.
        """
        self.machine.tds.switch_on_beam()
        LOG.debug(f"Beginning TDS scan measurement @ TDS ampl = {tds_amplitude}%")
        self.machine.tds.set_amplitude(tds_amplitude)
        time.sleep(self.SLEEP_AFTER_TDS_SETTING)
        scan_df = self.photographer.take_data(
            bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
        )

        return SetpointSnapshots(
            scan_df, "tds", dispersion_setpoint=dispersion_setpoint, measured_dispersion=measured_dispersion
        )

    def dispersion_scan(self, *, bg_shots: int, beam_shots: int) -> list[Path]:  # initial_dispersion=? to save time..
        """Do the dispersion scan part of the energy spread measurement"""
        LOG.info("Setting up dispersion scan")
        dispersion, dispersion_unc = self.reset_and_measure_dispersion()

        LOG.info(
            f"Starting dispersion scan: dispersions={self.dscan_config.dispersions}," f" {bg_shots=}, {beam_shots=}"
        )
        filenames = []
        self.machine.tds.switch_on_beam()
        for qsetting in self.dscan_config.scan_settings:
            snapshots = self.dispersion_scan_one_measurement(qsetting, bg_shots=bg_shots, beam_shots=beam_shots)
            fname = self.save_setpoint_snapshots(snapshots)
            filenames.append(fname)

        LOG.info(f"Finished dispersion scan: output pcl: {filenames=}")
        return filenames

    def reset_and_measure_dispersion(self) -> tuple[float, float]:
        """Reset the TDS amplitude and quadrupoles to their reference settings."""
        self.set_reference_tds_amplitude()
        return self.set_quads_and_get_dispersion(self.dscan_config.reference_setting)

    def dispersion_scan_one_measurement(self, quad_setting: QuadrupoleSetting, *, bg_shots: int, beam_shots: int):
        """Do a single dispersion scan measurement."""
        LOG.debug(f"Beginning dispersion scan measurement @ D = {quad_setting.dispersion}m")
        dispersion, dispersion_unc = self.set_quads_and_get_dispersion(quad_setting)

        self.machine.tds.switch_on_beam()
        scan_df = self.photographer.take_data(
            bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
        )
        return SetpointSnapshots(
            scan_df, "dispersion", dispersion, quad_setting.dispersion, (dispersion, dispersion_unc)
        )

    def set_quads_and_get_dispersion(self, quadrupole_setting: QuadrupoleSetting) -> tuple[float, float]:
        LOG.info(f"Setting dispersion for quad setting: at intended dispersion = {quadrupole_setting}.")
        self._apply_quad_setting(quadrupole_setting)
        return self.dispersion_measurer.measure()

    def set_reference_tds_amplitude(self) -> None:
        """Set the TDS amplitude to the reference value, i.e. the
        value used for the dispersion scan."""
        refampl = self.tds_config.reference_amplitude
        LOG.info(f"Setting reference TDS amplitude: {refampl}")
        self.machine.tds.set_amplitude(refampl)
        time.sleep(self.SLEEP_AFTER_TDS_SETTING)

    def set_reference_quads(self) -> None:
        """Set the full set of quadrupoles including the upstream matching ones."""
        LOG.info("Applying reference quadrupole settings to machine")
        self._apply_quad_setting(self.dscan_config.reference_setting)

    def _apply_quad_setting(self, setting: QuadrupoleSetting) -> None:
        """Apply an individual QuadrupoleSetting consisting of one or
        more quadrupole names, with k1ls, to the machine."""
        for name, k1l in zip(setting.names, setting.k1ls):
            k1l *= 1e3
            LOG.info("Setting quad: {name} to {k1l}")
            self.machine.set_quad(name, k1l)
        time.sleep(self.SLEEP_AFTER_QUAD_SETTING)

    def make_snapshots_filename(self, snapshot):
        """Make a human readnable name for the output dataframe"""
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        ampl = self.machine.tds.read_sp_amplitude()
        dispersion = snapshot.dispersion_setpoint
        scan_type = snapshot.scan_type
        fname = f"{timestamp}>>{scan_type}>>D={dispersion},TDS={ampl}%.pcl"
        return self.abs_output_directory() / fname

    def abs_output_directory(self) -> Path:
        return self.outpath.resolve()

    def save_setpoint_snapshots(self, setpoint_snapshot: SetpointSnapshots) -> str:
        fname = self.make_snapshots_filename(setpoint_snapshot)
        fname.parent.mkdir(exist_ok=True, parents=True)
        with fname.open("wb") as f:
            pickle.dump(setpoint_snapshot, f)
        LOG.info(f"Wrote measurement SetpointSnapshots (of {len(setpoint_snapshot)} snapshots) to: {fname}")
        return fname

    # def self_update_progress_file(self, scan_type: str, scan_setpoint: S, pcl_filename):
    #     from IPython import embed

    #     embed()


class ScreenPhotographer:
    def __init__(self, mps=None, machine=None):
        self.mps = mps if mps is not None else MPS()
        self.machine = machine
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

    def take_background_images(self, nshots: int, *, delay: float = 1):
        LOG.info("Taking background")
        self.switch_beam_off()
        time.sleep(delay)
        snapshots = []
        for i in range(nshots):
            LOG.info(f"Background snapshot: {i} / {nshots - 1}")
            snapshots.append(self.take_machine_snapshot(check_if_online=False))
            time.sleep(delay)
        LOG.info("Finished taking background")
        self.switch_beam_on()
        return pd.concat(snapshots)

    def take_beam_images(self, total_shots: int, *, delay: float = 1) -> pd.DataFrame:
        LOG.info("Taking beam images")
        self.switch_beam_on()
        snapshots = []

        ishot = 0
        while ishot < total_shots:
            LOG.info(f"Taking snapshot {ishot} / {total_shots - 1}")
            snapshots.append(self.take_machine_snapshot(check_if_online=True))

            # Check if the snapshot failed and we need to repeat the
            # snapshot procedure.
            if self.is_machine_offline():
                LOG.info(f"Failed snapshot {ishot} / {total_shots}: machine offline")
                self.switch_beam_on()
                time.sleep(DELAY_AFTER_BEAM_OFF)
                continue
            self.machine.tds.switch_on_beam()

            time.sleep(0.2)
            if not self.machine.tds.is_on_beam():
                self.machine.tds.switch_on_beam()
                time.sleep(0.1)
                print(f"is now on beam? {self.machine.tds.is_on_beam()}")

            ishot += 1

        return pd.concat(snapshots)

    def take_data(self, *, bg_shots: int, beam_shots: int, delay: float = 1) -> pd.DataFrame:
        bg = self.take_background_images(bg_shots, delay=delay)
        self.machine.tds.switch_on_beam()
        data = self.take_beam_images(beam_shots, delay=delay)
        return pd.concat([bg, data])

    def take_machine_snapshot(self, check_if_online=True) -> pd.DataFrame:
        LOG.debug("Taking machine snapshot")
        return self.machine.get_machine_snapshot(check_if_online=check_if_online)

    def switch_beam_off(self) -> None:
        LOG.debug("Switching beam off")
        self.mps.beam_off()

    def switch_beam_on(self) -> None:
        LOG.debug("Switching beam ON")
        self.mps.beam_on()

    def is_machine_offline(self) -> bool:
        return not self.machine.is_machine_online()


class BaseDispersionMeasurer:
    pass


class DispersionMeasurer(BaseDispersionMeasurer):
    def __init__(self, a1_voltages: list[float], machine):
        self.a1_voltages = a1_voltages
        self.machine = machine
        # if machine is None:
        #     self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

    def measure(self):
        raise NotImplementedError()

    # def measure(self, debug_path=None):
    #     LOG.info("Starting dispersion measurement, using A1 voltages: {self.a1_voltages}")
    #     centres = []
    #     energies = []
    #     for voltage in a1_voltages:
    #         self.machine.set_a1_voltage(voltage)
    #         x, _ = self._image_centre_of_mass()
    #         beam_energy = self.machine.get_beam_energy()
    #         centres.append(x)
    #         energies.append(self.machine.get_beam_energy())

    #     centres = [x * PIXEL_SCALE_X_M for x in centres]
    #     errors = np.ones_like(centres) * PIXEL_SCALE_X_M * 0.5
    #     _, (m, m_err) = linear_fit(energies, centres, PIXEL_SCALE_X_M)
    #     energy = 130
    #     dispersion = m * energy

    # def _image_centre_of_mass(self):
    #     image = self.machine.get_screen_image()
    #     y, x = ndi.center_of_mass(image)
    #     return x, y


class BasicDispersionMeasurer(BaseDispersionMeasurer):
    def measure(self) -> tuple[float, float]:
        dispersion = _repeat_float_input_until_valid("Enter dispersion in m: ")
        dispersion_unc = _repeat_float_input_until_valid("Enter dispersion unc in m: ")
        return dispersion, dispersion_unc


def _repeat_float_input_until_valid(prompt):
    while True:
        given = input(prompt)
        try:
            dispersion = float(given)
        except ValueError:
            print(f"Invalid dispersion: {given=}, go again")
            continue
        else:
            return dispersion


class TDS:
    RB_SP_TOLERANCE = 0.02
    def __init__(self):
        self.mi = XFELMachineInterface()
        self.bunch_one_timing = self.mi.get_value(self.BUNCH_ONE)

    def set_amplitude(self, amplitude: float) -> None:
        """Set the TDS amplitude"""
        LOG.debug(f"Setting TDS amplitude: {self.AMPLITUDE_SP} @ {amplitude}")
        self.mi.set_value(self.AMPLITUDE_SP, amplitude)

    def read_rb_amplitude(self) -> float:
        """Read back the TDS amplitude"""
        result = self.mi.get_value(self.AMPLITUDE_RB)
        LOG.debug(f"Reading TDS amplitude: {self.AMPLITUDE_RB} @ {result}")
        return result

    def read_sp_amplitude(self) -> float:
        """Read back the TDS amplitude"""
        result = self.mi.get_value(self.AMPLITUDE_SP)
        LOG.debug(f"Reading TDS amplitude: {self.AMPLITUDE_RB} @ {result}")
        return result

    def set_phase(self, phase: float) -> None:
        LOG.debug(f"Setting TDS amplitude: {self.PHASE_SP} @ {phase}")
        self.mi.set_value(self.PHASE_SP, phase)

    def read_rb_phase(self) -> float:
        result = self.mi.get_value(self.PHASE_RB)
        LOG.debug(f"Reading TDS amplitude: {self.PHASE_RB} @ {result}")
        return result

    def read_on_beam_timing(self):
        return self.mi.get_value(BUNCH_ONE)

    def is_powered(self) -> bool:
        LOG.debug("Checking if TDS is powered")
        rb = self.read_rb_amplitude()
        sp = self.read_sp_amplitude()
        relative_difference = abs(rb - sp) / sp
        powered = relative_difference < self.RB_SP_TOLERANCE
        LOG.debug(f"TDS RB ampl = {rb}; TDS SP = {sp}: {relative_difference=} -> {powered}")
        return powered

    def read_timing(self):
        return self.mi.get_value(self.EVENT)[2]

    def read_on_beam_timing(self):
        return self.mi.get_value(self.BUNCH_ONE)

    def is_on_beam(self) -> bool:
        LOG.debug("Checking if TDS is on beam")
        return self.read_timing() == self.read_on_beam_timing()

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
        self.mi.set_value(self.EVENT, on_data)


class I1TDS(TDS):
    AMPLITUDE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"
    AMPLITUDE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"
    PHASE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/PHASE.SAMPLE"
    PHASE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.PHASE"
    EVENT = EVENT10_CHANNEL
    BUNCH_ONE = BUNCH_ONE_TDS_I1


class B2TDS(TDS):
    AMPLITUDE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.AMPL"
    AMPLITUDE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSB2/AMPL.SAMPLE"
    EVENT = EVENT12_CHANNEL
    BUNCH_ONE = BUNCH_ONE_TDS_B2


class EnergySpreadMeasuringMachine(Machine):
    A1_VOLTAGE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
    A1_VOLTAGE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE"

    SCREEN_GAIN_CHANNEL = "!__placeholder__!"

    def __init__(self, snapshot):
        super().__init__(snapshot)
        self.tds = self.TDSCLS()

    def set_quad(self, name: str, value: float) -> None:
        """Set a particular quadrupole given by its name to the given value."""
        channel = f"XFEL.MAGNETS/MAGNET.ML/{name}/KICK_MRAD.SP"
        LOG.debug(f"Setting {channel} @ {value}")
        self.mi.set_value(channel, value)

    def set_a1_voltage(self, voltage: float) -> float:
        """Set the A1 voltage (used for dispersion measurements)"""
        LOG.debug(f"Setting A1 voltage: {self.A1_VOLTAGE_SP} @ {voltage}")
        self.mi.set_value(self.A1_VOLTAGE_SP, voltage)

    def read_a1_voltage(self) -> float:
        """Read back the A1 voltage"""
        result = self.mi.get_value(self.A1_VOLTAGE_RB)
        LOG.debug(f"Reading A1 voltage: {self.A1_VOLTAGE_RB} @ {result}")
        return result

    def get_screen_image(self):
        """Get screen image"""
        channel = self.SCREEN_CHANNEL
        LOG.debug(f"Reading image from {channel}")
        return self.mi.get_value(channel)

    def set_screen_gain(self, gain) -> None:
        """Set screen gain"""
        LOG.debug(f"Setting screen gain: {gain}")
        raise NotImplementedError
    

class TDSCalibratingMachine(Machine):
    def __init__(self, outdir):
        self.outdir = outdir
        self.tds = self.TDSCLS()
        self.mi = XFELMachineInterface()
        self.nphase_points = 60

    def calibrate(self, amplitudes, dirout=None):
        for amplitude in amplitudes:
            slope = self.get_slope(amplitude, npoints, dirout)

    def scan_phase(self):
        npoints = len(self.phase)

        for i, _ in enumerate(self.phases):
            self.tds.set_phase(phase)
            screen = self.get_screen_image()

    def get_slope(self, amplitude):
        # phases = np.linspace(-200, 200, 100)
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


class I1TDSCalibratingMachine(TDSCalibratingMachine):
    TDSCLS = I1TDS
    SCREEN_CHANNEL = I1D_SCREEN_ADDRESS


class B2TDSCalibratingMachine(TDSCalibratingMachine):
    TDSCLS = I1TDS
    SCREEN_CHANNEL = B2D_SCREEN_ADDRESS
    

class I1DEnergySpreadMeasuringMachine(EnergySpreadMeasuringMachine):
    SCREEN_CHANNEL = I1D_SCREEN_ADDRESS
    TEMPLATE_FN = make_injector_snapshot_template
    TDSCLS = I1TDS

class B2DEnergySpreadMeasuringMachine(EnergySpreadMeasuringMachine):
    SCREEN_CHANNEL = B2D_SCREEN_ADDRESS
    TEMPLATE_FN = make_b2d_snapshot_template
    TDSCLS = B2TDS


class ScanType(Enum):
    DISPERSION = auto()
    TDS = auto()
    BETA = auto()

    @classmethod
    @property
    def ALT_NAME_MAP(cls):
        return {cls.DISPERSION: "dscan", cls.TDS: "tscan", cls.BETA: "bscan"}

    def alt_name(self):
        return self.ALT_NAME_MAP[self]


@dataclass
class SetpointSnapshots:
    """Class for representing a set of snapshots at a single machine
    setpoint.  On top of the data read directly from the machine (the
    pd.DataFrame), there's also the scan_type, the dispersion setpoint
    and the corresponding measured dispersion.

    """

    snapshots: pd.DataFrame
    scan_type: ScanType
    dispersion_setpoint: float
    measured_dispersion: tuple[float, float]
    beta: float = None

    def __len__(self):
        return len(self.snapshots)

    def timestamped_name(self) -> str:
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        ampl_sp = _tds_amplitude_setpoint_from_df(self.snapshots)
        dispersion = self.dispersion_setpoint
        scan_name = self.scan_type.alt_name()
        return f"{timestamp}>>{scan_name}>>D={dispersion},TDS={ampl_sp}%.pcl"

    @property
    def tds_amplitude_setpoint(self) -> float:
        setpoints = self.snapshots[TDS_AMPLITUDE_READBACK_ADDRESS]
        one_setpoint = setpoints.iloc[0]

        if not (setpoints == one_setpoint).all():
            raise ValueError("Setpoint is not consistent across snapshots")
        return one_setpoint

    def __repr__(self):
        tname = type(self).__name__
        dx0 = self.dispersion_setpoint
        try:
            dx1, dx1e = self.measured_dispersion
        except TypeError:
            dx1 = self.measured_dispersion
            dx1e = 0.0
        bstring = ", beta={self.beta}" if self.beta else ""

        try:
            sname = self.scan_type.name
        except AttributeError:
            sname = self.scan_type

        out = (f"<{tname}: scan_type={sname}, dxm=({dx1}±{dx1e})m,"
               f" nsnapshots={len(self)}{bstring}>")
        return out

    def resolve_image_path(self, dirname):
        paths = self.snapshots["XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"]
        self.snapshots["XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"] = dirname / paths

def resume_from_output_directory(dirname: Union[os.PathLike, str]):
    """Resume a measurement from the output directory"""
    LOG.info(f"Trying to resume measurement in {dirname}")

    fname = dirname / "scan.toml"
    tconf = TDSScanConfiguration.from_config_file(fname)
    dconf = DispersionScanConfiguration.from_config_file(fname)


    # What we want, judging by the scan.toml
    tscan_amplitudes = set(tconf.scan_amplitudes)
    dscan_dispersions = set(s.dispersion for s in dconf.scan_settings)
    LOG.info(f"In scan.toml (desired machine setpoints): TDS scan amplitudes:"
             f" {tscan_amplitudes}, dispersion scan dispersions: {dscan_dispersions}")

    # What we have, judging by the output.
    for fpcl in dirname.glob("*.pcl"):
        with fpcl.open("rb") as f:
            snapshots = pickle.load(fpcl)

            # if not _is_complete_setpoint_measurement(snapshots, nbg, nbeam):
            #     continue

            tscan_amplitudes.discard(snapshots.tds_amplitude_setpoint)
            dscan_amplitudes.discard(snapshots.dispersion_setpoint)


def _tds_amplitude_setpoint_from_df(df):
    key_name = TDS.AMPLITUDE_RB
    col = df[key_name]
    value = col.iloc[0]
    if not (value == col).all():
        raise MalformedSnapshotDataFrame(f"{key_name} in {df} should be constant but is not")
    return value

def _is_complete_setpoint_measurement(snapshot, nbg, nbeam):
    # Given the generally >1 len pd.DataFrame, did i take all the data here?
    # actually I just redo it right?  If it was written, it worked!  This function will only have meaning when I eventually have some way to handle interruprs.
    # If the file exists, it's fine!
    from IPython import embed; embed()




# here is the AUTOGAIN on/off address for OTRC.55.I1 camera (as an
# example) - "XFEL.DIAG/CAMERA/OTRC.58.I1/GAINAUTO.NUM". There are 3
# options: 0 - OFF, 1 - Once, 2 - Continuous. You may set it either to
# 1 before each of your measurement series or to 2 and then it would
# be adjusted more or less at each image. The readout address (also
# OTRC.55.I1 example) - "XFEL.DIAG/CAMERA/OTRC.58.I1/GAINRAW". I
# believe it is all you need. Also, for instance, mAtthias as I know
# uses the AUTOGAIN in his emmitance measurement tool and it works
# fine. Though I don't know which exactly Once or Continuous one.
