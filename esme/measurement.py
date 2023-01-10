# Priority:

# Should keep track of good images/bad images.  Ignore bad ones.  Should already be possible?  using TDS ampl rb, etc...
# Should check TDS is on, check beam is on, check screen is out.


# Checklist:

# Do the dispersion measurement
# Image processing.
# Make sure it is fully mocked and tested

# Do the TDS Scan
# Automatically change the TDS voltage
# Make sure it is fully mocked and tested

# Do the Dispersion scan
# Automatically change quad strengths
# Make sure it is fully mocked and tested

# Misc:
# Writ metadata: measured dispersion, measured beta_x (if we do that)
# Detect when there is no TDS RF power.
# Detect when the TDS is on or off beam.
# Handle the TDS turning on/off randomly.
# Handle the beam turning off randomly.
# Check screen is out.
# Autogain for images.

# Dream:
# Automatic TDS Calibration (one day).

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import toml
import pandas as pd

from esme.mint.machine import MPS, Machine
from esme.injector_channels import SNAPSHOT_TEMPL
from esme.maths import line


LOG = logging.getLogger(__name__)

DELAY_AFTER_BEAM_OFF = 5


I1_DUMP_SCREEN_ADDRESS: str = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"
SNAPSHOT_TEMPL.add_image(I1_DUMP_SCREEN_ADDRESS, folder="./tds_images")

PIXEL_SCALE_X_M: float = 13.7369e-6
PIXEL_SCALE_Y_M: float = 11.1756e-6



class EnergySpreadMeasurementError(RuntimeError):
    pass


class MeasurementPrompt:
    pass


class EnergySpreadMeasurement:
    pass


@dataclass
class QuadrupoleSetting:
    names: list
    strengths: list
    dispersion: float

    def __post_init__(self): # TODO: test this!
        if len(self.names) != len(self.strengths):
            raise ValueError("Length mismatch between names and strengths")


@dataclass
class DispersionScanConfiguration:
    reference_setting: QuadrupoleSetting
    scan_settings: list[QuadrupoleSetting]

    @classmethod
    def from_conf(cls, config_path):
        # dscan_quads = []
        conf = toml.load(config_path)
        quads = conf["quads"]

        ref = quads["reference_optics"]
        reference_setting = QuadrupoleSetting(ref["names"],
                                              ref["strengths"],
                                              ref["dispersion"])

        scan_names = quads["scan_names"]
        scan_settings = []
        dscan = quads["dscan"]
        for settings in dscan.values():
            scan_settings.append(QuadrupoleSetting(scan_names,
                                                   settings["strengths"],
                                                   settings["dispersion"]))
        return cls(reference_setting, scan_settings)


@dataclass
class TDSScanConfiguration:
    reference_amplitude: float
    scan_amplitudes: list
    scan_dispersion: float

    @classmethod
    def from_conf(cls, config_path):
        conf = toml.load(config_path)
        tds = conf["tds"]
        return cls(reference_amplitude=tds["reference"],
                   scan_amplitudes=tds["scan_amplitude"],
                   scan_dispersion=tds["scan_dispersion"])


class MeasurementRunner:
    def __init__(self, basename, quad_config, tds_config, outdir="./", machine=None):
        self.basename = Path(basename)
        self.quad_config = quad_config
        self.tds_config = tds_config
        self.machine = machine
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)
        self.outdir = Path(outdir)

        self.dispersion_measurer = BasicDispersionMeasurer()

    def run(self):
        self.tds_scan(5, 30, delay=1)
        self.dispersion_scan(5, 30, delay=1)

    def tds_scan(self, bg_shots, data_shots, delay):
        """Do the TDS scan for this energy spread measurement"""
        LOG.info("Setting up TDS scan")

        self.set_reference_quads()

        LOG.info("Measing dispersion at start of TDS scan")
        dispersion, dispersion_unc = self.dispersion_measurer.measure()

        tds_amplitudes = self.tds_config.scan_amplitudes
        LOG.info(f"starting TDS scan: {tds_amplitudes=}, {bg_shots=}, {data_shots=}")

        for ampl in tds_amplitudes:
            self.machine.set_tds_amplitude(ampl)
            photographer = ScreenPhotographer()
            scan_df = photographer.take_data(bg_shots, data_shots, delay=delay)

            save_snapshot_df(scan_df, dispersion=dispersion, dispersion_unc=dispersion_unc)

    def dispersion_scan(self, bg_shots, data_shots, delay):
        """Do the dispersion scan for this energy spread measurement"""
        LOG.info("Setting up dispersion scan")

        self.set_reference_quads()

        ndispersions = len(self.quad_config.scan_settings)

        for i, setting in enumerate(self.quad_config.scan_settings):
            LOG.info(f"Starting dispersion scan measurement {i} / {ndispersions - 1}")
            design_dispersion = setting.dispersion
            self._apply_quad_setting(setting)
            dispersion, dispersion_unc = self.dispersion_measurer.measure()

            photographer = ScreenPhotographer()
            scan_df = photographer.take_data(bg_shots, data_shots, delay=delay)

            save_snapshot_df(scan_df, dispersion=dispersion, dispersion_unc=dispersion_unc)

    def beta_scan(self, bg_shots, data_shots):
        pass

    def set_reference_tds_amplitude(self):
        """Set the TDS amplitude to the reference value, i.e. the
        value used for the dispersion scan."""
        refampl = self.tds_config.reference_amplitude
        LOG.info(f"Setting reference TDS amplitude: {refampl}")
        self.machine.set_tds_amplitude(refampl)

    def set_reference_quads(self):
        """Set the full set of quadrupoles including the upstream matching ones."""
        LOG.info("Applying reference quadrupole settings to machine")
        self._apply_quad_setting(self.quad_config.reference_setting)

    def set_quads(self, dispersion, measure=True):
        LOG.info(f"Setting quads for dispersion @ {dispersion}")
        try:
            quad_setpoints = self.quad_config[dispersion]
        except KeyError:
            raise EnergySpreadMeasuringMachine("Unknown quad setting @ D={dispersion}")
        self._apply_quad_setting(quad_setpoints)

    def _apply_quad_setting(self, setting: QuadrupoleSetting):
        """Apply an individual QuadrupoleSetting consisting of one or
        more quadrupole names, with strengths, to the machine."""
        for name, strength in zip(setting.names, setting.strengths):
            LOG.info("Setting quad: {name} to {strength}")
            self.machine.set_quad(name, strength)

    def make_df_file_name(self, tds_amplitude, dispersion):
        """Make a human readnable name for the output dataframe"""
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        fname = f"{timestamp}>>>D={dispersion},TDS={tds_amplitude}%.pcl"
        return self.outdir / self.basename / fname

class ScreenPhotographer:
    def __init__(self, mps=None, machine=None):
        self.mps = mps if mps is not None else MPS()
        self.machine = machine
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

    def take_background(self, nshots, delay=1):
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

    def take_beam_images(self, total_shots, delay=1):
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
                LOG.info(f"Failed snapshot {ishot} / {total_shots}")
                snapshots = pop_last_snapshot(df)  # ? just add tds status to the df?
                self.switch_beam_on()
                time.sleep(DELAY_AFTER_BEAM_OFF)
                continue

            ishot += 1

        return pd.DataFrame(snapshots)

    def take_data(self, bg_shots, beam_shots, delay=1):
        bg = self.take_background(bg_shots, delay=delay)
        data = self.take_beam_images(data_shots, delay=delay)
        return pd.concat([bg, data])


    def take_machine_snapshot(self):
        LOG.info("Taking machine snapshot")
        return self.machine.get_machine_snapshot(check_if_online=True)

    def switch_beam_off(self):
        LOG.info("Switching beam off")
        self.mps.beam_off()

    def switch_beam_on(self):
        LOG.info("Switching beam ON")
        self.mps.beam_on()

    def is_machine_offline(self):
        return not self.machine.is_machine_online()

# class DispersionMeasurer:
#     def __init__(self, a1_voltages, machine=None):
#         self.a1_voltages = a1_voltages
#         if machine is None:
#             self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

#     def measure(self, debug_path=None):
#         LOG.info("Starting dispersion measurement, using A1 voltages: {self.a1_voltages}")
#         centres = []
#         energies = []
#         for voltage in a1_voltages:
#             self.machine.set_a1_voltage(voltage)
#             x, _ = self._image_centre_of_mass()
#             beam_energy = self.machine.get_beam_energy()
#             centres.append(x)
#             energies.append(self.machine.get_beam_energy())

#         centres = [x * PIXEL_SCALE_X_M for x in centres]
#         errors = np.ones_like(centres) * PIXEL_SCALE_X_M * 0.5
#         _, (m, m_err) = linear_fit(energies, centres, PIXEL_SCALE_X_M)
#         dispersion = m *

#     def _image_centre_of_mass(self):
#         image = self.machine.get_screen_image()
#         y, x = ndi.center_of_mass(image)
#         return x, y


class BasicDispersionMeasurer:
    def measure(self):
        dispersion = float(input("Enter dispersion in m:"))
        dispersion_unc = float(input("Enter dispersion unc in m:"))
        return dispersion, dispersion_unc


def handle_sigint():
    pass


def save_snapshot_df(df, fname, dispersion=None, dispersion_unc=None):
    LOG.info("Saving snapshot to df: {fname} with metadata: {metadata}")
    # df.attrs = metadata
    if dispersion is not None:
        df["dispersion"] = dispersion
    if dispersion_unc is not None:
        df["dispersion_uncertainty"] = dispersion_unc

    df.to_pickle(fname)


class EnergySpreadMeasuringMachine(Machine):
    QUAD_FACILITY = "XFEL.MAGNETS"
    QUAD_DEVICE = "MAGNET.ML"
    QUAD_PROPERTY = "KICK_MRAD.SP"

    TDS_AMPLITUDE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"
    # ? instead ? "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.POWER"
    TDS_AMPLITUDE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"

    A1_VOLTAGE_SP = "XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL"
    A1_VOLTAGE_RB = "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE"

    # # # For TDS I guess ?  Not sure
    # TIME_EVENT10 = "XEL.SDTIMER.CENTRAL/MASTER/EVENT10TL"

    # TDS_CONTROL = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL"

    SCREEN_CHANNEL = I1_DUMP_SCREEN_ADDRESS
    SCREEN_GAIN_CHANNEL = "!__placeholder__!"

    def __init__(self, snapshot):
        super().__init__(self, snapshot)

    def set_quad(self, name, value):
        """Set a particular quadrupole given by its name to the given value."""
        channel = f"{self.QUAD_FACILITY}/{self.QUAD_DEVICE}/{name}/{self.QUAD_PROPERTY}"
        LOG.debug(f"Setting {channel} @ {value}")
        self.mi.set_value(channel, value)

    def set_tds_amplitude(self, amplitude):
        LOG.debug(f"Setting TDS amplitude: {self.A1_VOLTAGE_SP} @ {voltage}")
        self.mi.set_value(self.TDS_AMPLITUDE_SP, amplitude)

    def read_tds_amplitude(self):
        result = self.mi.get_value(self.TDS_AMPLITUDE_RB)
        LOG.debug(f"Reading TDS amplitude: {self.TDS_AMPLITUDE_RB} @ {result}")
        return result

    def set_a1_voltage(self, voltage):
        LOG.debug(f"Setting A1 voltage: {self.A1_VOLTAGE_SP} @ {voltage}")
        self.mi.set_value(self.A1_VOLTAGE_SP, voltage)

    def read_a1_voltage(self):
        result = self.mi.get_value(self.A1_VOLTAGE_RB)
        LOG.debug(f"Reading A1 voltage: {self.A1_VOLTAGE_RB} @ {result}")
        return result

    def turn_tds_off_beam(self):
        LOG.debug(f"Setting TDS off beam")
        self._switch_tds_on_off_beam(on=False)

    def turn_tds_on_beam(self):
        LOG.debug(f"Setting TDS on beam")
        self._switch_tds_on_off_beam(on=True)

    def _switch_tds_on_off_beam(self, *, on: bool):
        bunch_number = 1
        on_data = [bunch_number, int(on), 0, 0]  # 3rd: "kicker", 4th: "WS-subtrain"

        self.mi.set_value("XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL", on_data)
        # "How many pulses to kick" (???)  Not sure why 1000 in particular
        temp = self.mi.set_value('XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/PULSES.ACTIVE', 1000)
        time.sleep(0.1)
        # "Start kicking"
        temp = self.mi.set_value('XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/START', 1)
        time.sleep(0.2)

    def get_screen_image(self):
        channel = self.SCREEN_CHANNEL
        LOG.debug(f"Reading image from {channel}")
        return self.mi.get_value(channel)

    def set_screen_gain(self, gain):
        LOG.debug("Setting screen gain: {gain}.")
        pass
