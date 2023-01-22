# Priority:

# Should keep track of good images/bad images.  Ignore bad ones.  Should already be possible?  using TDS ampl rb, etc...
# Should check TDS is on, check beam is on, check screen is out.


# Checklist:

# Do the dispersion measurement
# Image processing.
# Make sure it is fully mocked and tested

# Do the TDS Scan
# Automatically change the TDS voltage: DONE
# Make sure it is fully mocked and tested

# Do the Dispersion scan
# Automatically change quad strengths: DONE
# Make sure it is fully mocked and tested

# Interrupt handler
# Restore measurement process so don't always have to start from scratch!
# How about a file that goes in the output directory that stores progress?
# And then I read that back in and try and resume.
# (also stores scan.toml used path etc... or indeed maybe just copy this to the destination?).

# Copy the scan.toml to the output directory.  do a set difference to find what is missing.
# maybe there are also images missing.

# only need a true sig int handler I think when it's a GUI (interrupt
# current measurement), can go again, or do "new measurement" button.

# Do I even really need a fancy sigint handler if I just simply repeat
# any measurement that didn't perfectly complete?

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

# Dream:
# Automatic TDS Calibration (one day).

# Questions for Bolko:
# Is EVENT10_CHANNEL and TDS_ON_BEAM_EVENT10 in injector_channel.py correct?  Good.
# How to tell if TDS RF power is on?  Check Rb.
# How to set gain?  ?
# How long to sleep for quads to adjust? ?
# How to tell if the screen is out? Done.

# Nice to have one day:
# Warn if laser heater is on.
# Warn if AH1 is on.

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

from esme.mint.machine import MPS, Machine
from esme.injector_channels import SNAPSHOT_TEMPL, DUMP_SCREEN_ADDRESS, TDS_AMPLITUDE_READBACK_ADDRESS

TDSScanConfigurationSelf = TypeVar("TDSScanConfigurationSelfType", bound="TDSScanConfiguration")
DispersionScanConfigurationSelf = TypeVar("DispersionScanConfigurationSelfType", bound="DispersionScanConfiguration")


LOG = logging.getLogger(__name__)

DELAY_AFTER_BEAM_OFF: float = 5.0


SCREEN_NAME: Path = Path("XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ").parent.name
SNAPSHOT_TEMPL.add_image(DUMP_SCREEN_ADDRESS, folder=f"./images-{SCREEN_NAME}")

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

    """

    names: list
    strengths: list
    dispersion: float

    def __post_init__(self):  # TODO: test this!
        if len(self.names) != len(self.strengths):
            raise ValueError("Length mismatch between names and strengths")


@dataclass
class DispersionScanConfiguration:
    """A series of QuadrupoleSetting instances for each datapoint in
    the dispersion scan, as well as the "reference_setting"---used in
    the TDS scan and set at the start of the dispersion scan."""

    reference_setting: QuadrupoleSetting
    scan_settings: list[QuadrupoleSetting]

    @classmethod
    def from_config_file(cls, config_path: os.PathLike) -> DispersionScanConfigurationSelf:
        # dscan_quads = []
        conf = toml.load(config_path)
        quads = conf["quads"]

        ref = quads["reference_optics"]
        reference_setting = QuadrupoleSetting(ref["names"], ref["strengths"], ref["dispersion"])

        scan_settings = []
        dscan = quads["dscan"]
        dscan_quad_names = dscan["names"]
        for dispersion, scan_strengths in zip(dscan["dispersions"], dscan["strengths"]):
            scan_settings.append(QuadrupoleSetting(dscan_quad_names, scan_strengths, dispersion))
        return cls(reference_setting, scan_settings)

    @property
    def dispersions(self) -> list[float]:
        return [qsetting.dispersion for qsetting in self.scan_settings]


@dataclass
class TDSScanConfiguration:
    """"""

    reference_amplitude: float
    scan_amplitudes: list
    scan_dispersion: float

    @classmethod
    def from_config_file(cls, config_path: os.PathLike) -> TDSScanConfigurationSelf:
        conf = toml.load(config_path)
        tds = conf["tds"]
        return cls(
            reference_amplitude=tds["reference_amplitude"],
            scan_amplitudes=tds["scan_amplitudes"],
            scan_dispersion=tds["scan_dispersion"],
        )


class MeasurementRunner:
    SLEEP_BETWEEN_SNAPSHOTS = 1
    SLEEP_AFTER_TDS_SETTING = 0.5
    SLEEP_AFTER_QUAD_SETTING = 0.5

    def __init__(
        self,
        name,
        dscan_config: DispersionScanConfiguration,
        tds_config: TDSScanConfiguration,
        outdir: Union[os.PathLike, str] = "./",
        machine: Optional[EnergySpreadMeasuringMachine] = None,
        mps: Optional[MPS] = None,
        dispersion_measurer: Optional[Type[BaseDispersionMeasurer]] = None,
    ):
        """name is used for the output file name"""
        self.name = name
        self.dscan_config = dscan_config
        self.tds_config = tds_config
        self.outdir = Path(outdir)

        self.machine = machine
        if self.machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

        self.photographer = ScreenPhotographer(machine=self.machine, mps=mps)
        self.dispersion_measurer = dispersion_measurer
        if self.dispersion_measurer is None:
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

        scan_df = self.photographer.take_data(
            bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
        )
        return SetpointSnapshots(
            scan_df, "dispersion", dispersion, quad_setting.dispersion, (dispersion, dispersion_unc)
        )

    def set_quads_and_get_dispersion(self, quadrupole_setting: QuadrupoleSetting) -> tuple[float, float]:
        LOG.info("Setting dispersion for quad setting: at intended dispersion = {}.")
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
        more quadrupole names, with strengths, to the machine."""
        for name, strength in zip(setting.names, setting.strengths):
            LOG.info("Setting quad: {name} to {strength}")
            self.machine.set_quad(name, strength)
        time.sleep(self.SLEEP_AFTER_QUAD_SETTING)

    def make_snapshots_filename(self, snapshot):
        """Make a human readnable name for the output dataframe"""
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        ampl = self.machine.read_tds_sp_amplitude()
        dispersion = snapshot.dispersion_setpoint
        scan_type = snapshot.scan_type
        fname = f"{timestamp}>>{scan_type}>>D={dispersion},TDS={ampl}%.pcl"
        return self.abs_output_directory() / fname

    def abs_output_directory(self) -> Path:
        return (self.outdir / self.name).resolve()

    def save_setpoint_snapshots(self, setpoint_snapshot: SetpointSnapshots) -> str:
        outdir = self.abs_output_directory()
        fname = self.make_snapshots_filename(setpoint_snapshot)
        fname.parent.mkdir(exist_ok=True, parents=True)
        with fname.open("rb") as f:
            pickle.dump(setpoint_snapshot, f)
        LOG.info(f"Wrote measurement SetpointSnapshots (of {len(setpoint_snapshot)} snapshots) to: {fname}")
        return fname

    def progress_file_name(self) -> Path:
        return self.name / "progress.toml"

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
            snapshots.append(self.take_machine_snapshot())
            time.sleep(delay)
        LOG.info("Finished taking background")
        self.switch_beam_on()
        return pd.DataFrame(snapshots)

    def take_beam_images(self, total_shots: int, *, delay: float = 1) -> pd.DataFrame:
        LOG.info("Taking beam images")
        self.switch_beam_on()
        snapshots = []

        ishot = 0
        while ishot < total_shots:
            LOG.info(f"Taking snapshot {ishot} / {total_shots - 1}")
            snapshots.append(self.take_machine_snapshot())

            # Check if the snapshot failed and we need to repeat the
            # snapshot procedure.
            if self.is_machine_offline():
                LOG.info(f"Failed snapshot {ishot} / {total_shots}: machine offline")
                self.switch_beam_on()
                time.sleep(DELAY_AFTER_BEAM_OFF)
                continue

            ishot += 1

        return pd.DataFrame(snapshots)

    def take_data(self, *, bg_shots: int, beam_shots: int, delay: float = 1) -> pd.DataFrame:
        bg = self.take_background_images(bg_shots, delay=delay)
        data = self.take_beam_images(beam_shots, delay=delay)
        return pd.concat([bg, data])

    def take_machine_snapshot(self) -> pd.DataFrame:
        LOG.debug("Taking machine snapshot")
        return self.machine.get_machine_snapshot(check_if_online=True)

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
    def __init__(self, a1_voltages: list[float], machine=None):
        self.a1_voltages = a1_voltages
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

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
        dispersion = float(input("Enter dispersion in m:"))
        dispersion_unc = float(input("Enter dispersion unc in m:"))
        return dispersion, dispersion_unc


def handle_sigint():
    pass


class TDS:
    RB_SP_TOLERANCE = 0.02
    AMPLITUDE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"
    AMPLITUDE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"
    EVENT = "XFEL.DIAG/TIMER.CENTRAL/MASTER/EVENT10"

    def __init__(self, mi=None):
        self.mi = mi

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

    def is_powered(self) -> bool:
        LOG.debug("Checking if TDS is powered")
        rb = self.read_rb_amplitude()
        sp = self.read_sp_amplitude()
        relative_difference = abs(rb - sp) / sp
        powered = relative_difference < self.RB_SP_TOLERANCE
        LOG.debug(f"TDS RB ampl = {rb}; TDS SP = {sp}: {relative_difference=} -> {powered}")
        return powered

    def is_on_beam(self) -> bool:
        pass

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

        self.mi.set_value("XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL", on_data)
        # "How many pulses to kick" (???)  Not sure why 1000 in particular
        self.mi.set_value('XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/PULSES.ACTIVE', 1000)
        time.sleep(0.1)
        # "Start kicking"
        self.mi.set_value('XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/START', 1)
        time.sleep(0.2)


class EnergySpreadMeasuringMachine(Machine):

    A1_VOLTAGE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
    A1_VOLTAGE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE"

    SCREEN_CHANNEL = DUMP_SCREEN_ADDRESS
    SCREEN_GAIN_CHANNEL = "!__placeholder__!"

    def __init__(self, snapshot):
        super().__init__(snapshot)
        self.tds = TDS(self.mi)

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


# def get_progress_from_output_dir(dname):
#     # Load the scan.toml in the output directory
#     fscan = toml.load(Path(fname) / "scan.toml")

#     # These are the desired completed scans
#     dconf = DispersionScanConfiguration.from_config_file(fscan)
#     tconf = TDSScanConfiguration.from_config_file(fscan)

#     # Now load the output in the output directory and find out what
#     # we've actually got by looking at the TDS amplitude setpoints,  dispersion setpoints and sc


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
        return f"{timestamp}>>{scan_type}>>D={dispersion},TDS={ampl_sp}%.pcl"

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
        dx1, dx1e = self.measured_dispersion
        bstring = ", beta={self.beta}" if self.beta else ""
        out = (f"<{tname}: scan_type={self.scan_type.name}, {dx0=}, dxm=({dx1}±{dx1e})m,"
               f" nsnapshots={len(self)}{bstring}>")
        return out


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
             f" {tscan_amplitudes}, dispersion scan amplitudes: {dscan_amplitudes}")

    # What we have, judging by the output.
    for fpcl in dirname.glob("*.pcl"):
        with pcl.open("rb") as f:
            snapshots = pickle.load(fpcl)

            if not _is_complete_snapshot(snapshot, nbg, nbeam):
                continue

            tscan_amplitudes.discard(snapshots.tds_amplitude_setpoint)
            dscan_amplitudes.discard(snapshots.dispersion_setpoint)



def _is_complete_snapshot(snapshot):
    pass
