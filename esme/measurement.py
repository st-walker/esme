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

import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Optional, TypeVar, Union

import matplotlib.image
import numpy as np
import pandas as pd
import toml

from esme.channels import (
    BEAM_ALLOWED_ADDRESS,
    TDS_I1_AMPLITUDE_READBACK_ADDRESS,
    make_injector_snapshot_template,
)
from esme.mint import (  # , DictionaryXFELMachineInterface, TimestampedImage
    MPS,
    Machine,
    XFELMachineInterface,
)
from esme.tds import MockTDS

TDSScanConfigurationSelf = TypeVar(
    "TDSScanConfigurationSelfType", bound="TDSScanConfiguration"
)
DispersionScanConfigurationSelf = TypeVar(
    "DispersionScanConfigurationSelfType", bound="DispersionScanConfiguration"
)


LOG = logging.getLogger(__name__)

DELAY_AFTER_BEAM_OFF: float = 0.5

PIXEL_SCALE_X_M: float = 13.7369e-6
PIXEL_SCALE_Y_M: float = 11.1756e-6


# class MeasurementServer:
#     Send request, receive request.  process.  return payload.

# class MeasurementRequest:
#     pass

# class SnapshotRequest:
#     ...

# class EnergySpreadMeasurementError(RuntimeError):
#     pass


class TDSScanRunner:
    def __init__(self, tscan_config):
        pass


class DispersionScanRunner:
    pass


class MeasurementRunner:
    SLEEP_BETWEEN_SNAPSHOTS = 0.2
    # SLEEP_AFTER_TDS_SETTING = 0.5
    SLEEP_AFTER_QUAD_SETTING = 1

    def __init__(
        self,
        outdir: Union[os.PathLike, str],
        dscan_config: DispersionScanConfiguration,
        tds_config: TDSScanConfiguration,
        machine: Optional[EnergySpreadMeasuringMachine],
        mps: Optional[MPS] = None,
    ):
        """name is used for the output file name"""
        self.outdir = Path(outdir)
        self.dscan_config = dscan_config
        self.tds_config = tds_config

        self.machine = machine
        if self.machine is None:
            templ = make_injector_snapshot_template(self.abs_output_directory())
            self.machine = EnergySpreadMeasuringMachine(templ)

        self.snapshotter = Snapshotter(machine=self.machine, mps=mps, outdir=outdir)

    def run(self, *, bg_shots: int, beam_shots: int) -> None:
        tds_filenames = self.tds_scan(bg_shots=bg_shots, beam_shots=beam_shots)
        dispersion_filenames = self.dispersion_scan(
            bg_shots=bg_shots, beam_shots=beam_shots
        )
        return tds_filenames, dispersion_filenames

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
    ) -> SetpointMachineSnapshots:
        """Do a single tds scan measurement at the given TDS
        amplitude, with a variable number of background images
        (bg_shots), beam images (beam_shots) and a delay between each
        image being taken.
        """
        self.machine.tds.switch_on_beam()
        LOG.debug(f"Beginning TDS scan measurement @ TDS ampl = {tds_amplitude}%")
        self.machine.tds.set_amplitude(tds_amplitude)
        time.sleep(self.SLEEP_AFTER_TDS_SETTING)
        scan_df = self.snapshotter.take_data(
            bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
        )
        return SetpointMachineSnapshots(
            scan_df,
            ScanType.TDS,
            dispersion_setpoint=dispersion_setpoint,
            measured_dispersion=measured_dispersion,
            beta=0.6,
            screen_channel=self.machine.SCREEN_CHANNEL,
        )

    def dispersion_scan(
        self, *, bg_shots: int, beam_shots: int
    ) -> list[Path]:  # initial_dispersion=? to save time..
        """Do the dispersion scan part of the energy spread measurement"""
        LOG.info("Setting up dispersion scan")
        dispersion, dispersion_unc = self.reset_and_measure_dispersion()

        LOG.info(
            f"Starting dispersion scan: dispersions={self.dscan_config.dispersions},"
            f" {bg_shots=}, {beam_shots=}"
        )
        filenames = []
        self.machine.tds.switch_on_beam()
        for qsetting in self.dscan_config.scan_settings:
            snapshots = self.dispersion_scan_one_measurement(
                qsetting, bg_shots=bg_shots, beam_shots=beam_shots
            )
            fname = self.save_setpoint_snapshots(snapshots)
            filenames.append(fname)

        LOG.info(f"Finished dispersion scan: output pcl: {filenames=}")
        return filenames

    def reset_and_measure_dispersion(self) -> tuple[float, float]:
        """Reset the TDS amplitude and quadrupoles to their reference settings."""
        self.set_reference_tds_amplitude()
        return self.set_quads_and_get_dispersion(self.dscan_config.reference_setting)

    def dispersion_scan_one_measurement(
        self, quad_setting: QuadrupoleSetting, *, bg_shots: int, beam_shots: int
    ):
        """Do a single dispersion scan measurement."""
        LOG.debug(
            f"Beginning dispersion scan measurement @ D = {quad_setting.dispersion}m"
        )
        dispersion, dispersion_unc = self.set_quads_and_get_dispersion(quad_setting)

        self.machine.tds.switch_on_beam()
        scan_df = self.snapshotter.take_data(
            bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
        )
        return SetpointMachineSnapshots(
            scan_df,
            scan_type=ScanType.DISPERSION,
            dispersion_setpoint=quad_setting.dispersion,
            measured_dispersion=(dispersion, dispersion_unc),
            screen_channel=self.machine.SCREEN_CHANNEL,
            beta=0.6,
        )

    def set_quads_and_get_dispersion(
        self, quadrupole_setting: QuadrupoleSetting
    ) -> tuple[float, float]:
        LOG.info(
            f"Setting dispersion for quad setting: at intended dispersion = {quadrupole_setting}."
        )
        self._apply_quad_setting(quadrupole_setting)
        return quadrupole_setting.dispersion, 0

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
            LOG.info(f"Setting quad: {name} to {k1l}")
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
        return self.outdir.resolve()

    def save_setpoint_snapshots(
        self, setpoint_snapshot: SetpointMachineSnapshots
    ) -> str:
        fname = self.make_snapshots_filename(setpoint_snapshot)
        fname.parent.mkdir(exist_ok=True, parents=True)
        with fname.open("wb") as f:
            pickle.dump(setpoint_snapshot, f)
        LOG.info(
            f"Wrote measurement SetpointMachineSnapshots (of {len(setpoint_snapshot)} snapshots) to: {fname}"
        )
        return fname

    def measure_flat(self):
        pass

    def flat_dispersion_scan(self, dispersion_setpoint, nbg, nbeam):
        for qsetting in self.dscan_config.scan_settings:
            set_quads()
            for i in range(self.nbg):
                for payload in self.photographer.take_bg(nbg):
                    yield payload
            for i in range(self.nbeam):
                for payload in self.photographer.take_beam(nbg):
                    yield payload

    def flat_tds_scan(self, tds_percentage):
        pass


class DataTaker:
    SLEEP_BETWEEN_SNAPSHOTS = 0.2
    SLEEP_AFTER_TDS_SETTING = 0.5
    SLEEP_AFTER_QUAD_SETTING = 1

    def __init__(
        self,
        dscan_config: DispersionScanConfiguration,
        tds_config: TDSScanConfiguration,
        machine: EnergySpreadMeasuringMachine,
    ):
        """name is used for the output file name"""
        self.dscan_config = dscan_config
        self.tds_config = tds_config
        self.machine = machine
        self.snapshotter = SingleSnapshotter(
            machine=self.machine, mps=MockMPS(mi=self.machine.mi)
        )

    def snapshot_sleep(self):
        time.sleep(self.SLEEP_BETWEEN_SNAPSHOTS)

    def measure(self, *, bg_shots: int, beam_shots: int) -> None:
        yield from self.tds_scan(bg_shots=bg_shots, beam_shots=beam_shots)
        yield from self.dispersion_scan(bg_shots=bg_shots, beam_shots=beam_shots)

    def take_background(self, nshots, tds_amplitude, dispersion_setting, scan_type):
        snapshot, image = self.snapshotter.take_background_snapshot(
            switch_beam_off=True
        )
        yield MeasurementPayload(
            image, snapshot, tds_amplitude, dispersion_setting, True, scan_type
        )
        self.snapshot_sleep()
        for i in range(nshots - 1):
            snapshot, image = self.snapshotter.take_background_snapshot()
            yield MeasurementPayload(
                image, snapshot, tds_amplitude, dispersion_setting, True, scan_type
            )
            self.snapshot_sleep()

    def take_beam(self, nshots, tds_amplitude, dispersion_setting, scan_type):
        for i in range(nshots):
            snapshot, image = self.snapshotter.take_beam_snapshot()
            yield MeasurementPayload(
                image, snapshot, tds_amplitude, dispersion_setting, False, scan_type
            )
            self.snapshot_sleep()

    def tds_scan(self, *, bg_shots: int, beam_shots: int) -> list[Path]:
        """Do the TDS scan for this energy spread measurement"""
        LOG.info("Setting up TDS scan")

        dsetpoint = self.dscan_config.reference_setting.dispersion
        self.set_reference_quads()
        self.set_reference_tds_amplitude()

        tds_amplitudes = self.tds_config.scan_amplitudes

        LOG.info(f"starting TDS scan: {tds_amplitudes=}, {bg_shots=}, {beam_shots=}")
        for ampl in tds_amplitudes:
            self.machine.tds.set_amplitude(ampl)
            yield from self.take_background(
                bg_shots,
                ampl,
                self.dscan_config.reference_setting.dispersion,
                ScanType.TDS,
            )
            yield from self.take_beam(
                beam_shots,
                ampl,
                self.dscan_config.reference_setting.dispersion,
                ScanType.TDS,
            )

    def dispersion_scan(
        self, *, bg_shots: int, beam_shots: int
    ) -> list[Path]:  # initial_dispersion=? to save time..
        """Do the dispersion scan part of the energy spread measurement"""
        LOG.info("Setting up dispersion scan")
        self.set_reference_quads()
        self.set_reference_tds_amplitude()

        LOG.info(
            f"Starting dispersion scan: dispersions={self.dscan_config.dispersions},"
            f" {bg_shots=}, {beam_shots=}"
        )
        for qsetting in self.dscan_config.scan_settings:
            self.apply_quad_setting(qsetting)
            yield from self.take_background(
                bg_shots,
                self.tds_config.reference_amplitude,
                qsetting.dispersion,
                ScanType.DISPERSION,
            )
            yield from self.take_beam(
                beam_shots,
                self.tds_config.reference_amplitude,
                qsetting.dispersion,
                ScanType.DISPERSION,
            )

    def apply_quad_setting(self, setting: QuadrupoleSetting) -> None:
        """Apply an individual QuadrupoleSetting consisting of one or
        more quadrupole names, with k1ls, to the machine."""
        for name, k1l in zip(setting.names, setting.k1ls):
            k1l *= 1e3
            LOG.info(f"Setting quad: {name} to {k1l}")
            self.machine.set_quad(name, k1l)
        time.sleep(self.SLEEP_AFTER_QUAD_SETTING)

    # def dispersion_scan_one_measurement(self, quad_setting: QuadrupoleSetting, *, bg_shots: int, beam_shots: int):
    #     """Do a single dispersion scan measurement."""
    #     LOG.debug(f"Beginning dispersion scan measurement @ D = {quad_setting.dispersion}m")
    #     dispersion, dispersion_unc = self.set_quads_and_get_dispersion(quad_setting)

    #     self.machine.tds.switch_on_beam()
    #     scan_df = self.snapshotter.take_data(
    #         bg_shots=bg_shots, beam_shots=beam_shots, delay=self.SLEEP_BETWEEN_SNAPSHOTS
    #     )
    #     return SetpointMachineSnapshots(
    #         scan_df, "dispersion", dispersion, quad_setting.dispersion, (dispersion, dispersion_unc)
    #     )

    def set_quads_and_get_dispersion(
        self, quadrupole_setting: QuadrupoleSetting
    ) -> tuple[float, float]:
        LOG.info(
            f"Setting dispersion for quad setting: at intended dispersion = {quadrupole_setting}."
        )
        self._apply_quad_setting(quadrupole_setting)
        return quadrupole_setting.dispersion, 0.0

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
            LOG.info(f"Setting quad: {name} to {k1l}")
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
        return self.outdir.resolve()


@dataclass
class MeasurementPayload:
    image: TimestampedImage
    snapshot: pd.DataFrame
    tds_percentage: float
    dispersion_setpoint: float
    is_bg: bool
    scan_type: ScanType

    def __repr__(self):
        return f"MeasurementPayload: image:{self.image}, bg:{self.is_bg}, tds={self.tds_percentage}, disp={self.dispersion_setpoint}, scan_type:{self.scan_type}"


class SingleSnapshotter:
    def __init__(self, mps=None, machine=None):
        self.machine = machine
        self.mps = MPS(mi=self.machine.mi) if mps is None else mps
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)

    def take_background_snapshot(self, switch_beam_off=True):
        if switch_beam_off:
            self.mps.beam_off()
            time.sleep(DELAY_AFTER_BEAM_OFF)
        return self.take_machine_snapshot(check_if_online=False)

    def take_beam_snapshot(self):
        if not self.machine.tds.is_on_beam():
            self.machine.tds.switch_on_beam()

        result = self.take_machine_snapshot(check_if_online=True)
        print(result)
        # Should ideally still be on beam afterwards, just to be
        # sure...
        # Alternatively: I should check the dataframe says it's on beam...
        if not self.machine.tds.is_on_beam():
            return self.take_beam_image()
        return result

    def take_machine_snapshot(self, check_if_online=True) -> tuple[pd.DataFrame, list]:
        LOG.debug("Taking machine snapshot")
        df, image = self.machine.get_machine_snapshot(check_if_online=check_if_online)
        return df, image


class Snapshotter:
    def __init__(self, mps=None, machine=None, outdir=None):
        self.mps = mps if mps is not None else MPS()
        self.machine = machine
        if machine is None:
            self.machine = EnergySpreadMeasuringMachine(SNAPSHOT_TEMPL)
        if outdir is None:
            self.outdir = outdir

    def take_background_images(self, nshots: int, *, delay: float = 1):
        LOG.info("Taking background")
        self.switch_beam_off()
        time.sleep(2)
        snapshots = []
        for i in range(nshots):
            LOG.info(f"Background snapshot: {i} / {nshots - 1}")
            df, image = self.take_machine_snapshot(check_if_online=False)
            path = self.machine.snapshot.image_folders[0] / (image.name() + ".png")
            matplotlib.image.imsave(path, image.image)
            with path.with_suffix(".pcl").open("wb") as f:
                import pickle

                pickle.dump(image.image, f)

            snapshots.append(df)
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
            df, image = self.take_machine_snapshot(check_if_online=False)
            snapshots.append(df)
            path = self.machine.snapshot.image_folders[0] / (image.name() + ".png")
            matplotlib.image.imsave(path, image.image)
            with path.with_suffix(".pcl").open("wb") as f:
                import pickle

                pickle.dump(image.image, f)

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

    def take_data(
        self, *, bg_shots: int, beam_shots: int, delay: float = 1
    ) -> pd.DataFrame:
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


class EnergySpreadMeasuringMachine(Machine):
    def __init__(
        self,
        tds: TDS,
        template,
        screne_channel: str,
        outdir=None,
        mi=None,
    ):
        super().__init__(self.make_template(outdir))
        self.tds = self.TDSCLS(mi=mi)

    def set_quad(self, name: str, value: float) -> None:
        """Set a particular quadrupole given by its name to the given value."""
        channel = f"XFEL.MAGNETS/MAGNET.ML/{name}/KICK_MRAD.SP"
        LOG.debug(f"Setting {channel} @ {value}")
        self.mi.set_value(channel, value)

    def get_screen_image(self):
        """Get screen image"""
        channel = self.SCREEN_CHANNEL
        LOG.debug(f"Reading image from {channel}")
        return self.mi.get_value(channel)


class EnergySpreadMeasuringMachineReplayer(EnergySpreadMeasuringMachine):
    def __init__(self, outdir, anatoml):
        self.conf = toml.load(anatoml)

        data = self.conf["data"]

        bp = Path(data["basepath"])

        self.tscan = [pd.read_pickle(bp / p) for p in data["tscan"]["fnames"]]
        self.dscan = [pd.read_pickle(bp / p) for p in data["dscan"]["fnames"]]

        self.scans = np.array(self.tscan + self.dscan)
        self.scan_tds_amplitudes = np.array(
            [x.tds_amplitude_setpoint for x in self.scans]
        )
        self.scan_dispersion_setpoints = np.array(
            [x.dispersion_setpoint for x in self.scans]
        )

        # for scan in self.scans:
        #     from IPython import embed; embed()
        #     scan.snapshot[self.TDSCLS.EVENT] = scan.snapshot[self.TDSCLS.BUNCH_ONE]

        try:
            bunch_one = self.tscan[0].snapshots[I1TDS.BUNCH_ONE].iloc[0]
        except KeyError:
            # Assume always on beam.
            bunch_one = self.tscan[0].snapshots[I1TDS.EVENT].iloc[0][2]

        # from IPython import embed; embed()

        event_10 = self.tscan[0].snapshots[I1TDS.EVENT].iloc[0]

        self.mi = DictionaryXFELMachineInterface(
            {
                I1TDS.BUNCH_ONE: bunch_one,
                # Always online:
                "XFEL.DIAG/TOROID/TORA.60.I1/CHARGE.ALL": 1,
                I1TDS.EVENT: event_10,
            }
        )
        super().__init__(outdir, mi=self.mi)

        self.quads = {}
        self.dispersion_setpoint = 0

        self.tds = MockTDS(self, mi=self.mi)

        self.bg_measurement_number = 0
        self.beam_measurement_number = 0

    def _pick_snapshot_df(self):
        # First pick by TDS %.
        tds_amplitude = self.tds.read_rb_amplitude()
        # Only check the correct TDS setpoints:
        for scan in self.scans[self.scan_tds_amplitudes == float(tds_amplitude)]:
            df = scan.snapshots
            is_bg = not bool(self.mi.get_value(BEAM_ALLOWED_ADDRESS))

            if is_bg:
                df = df[df[BEAM_ALLOWED_ADDRESS] == False]
                index = self.bg_measurement_number
            else:
                df = df[df[BEAM_ALLOWED_ADDRESS] == True]
                index = self.beam_measurement_number

            # TDS Scan is easy to check but dispersion scan is tricky,
            # we do it by comparing this particular quadurpole which
            # changes a lot...
            # XXX!!!
            from math import isclose

            if isclose(df.iloc[index]["QI.64.I1"], self.quads["QI.64.I1"], abs_tol=1):
                return df.iloc[index]
        else:
            print("oh shit!!")
            x = "shite"
            from IPython import embed

            embed()

    def get_orbit(self, data, all_names):
        df = self._pick_snapshot_df()
        bpm_mask = df.keys().str.startswith("BPM")
        bpm_mask &= df.keys().str.endswith(".X") | df.keys().str.endswith(".Y")

        new_data = df[bpm_mask].values
        new_names = list(df[bpm_mask].keys())

        data = np.append(data, new_data)
        all_names = np.append(all_names, new_names)

        return data, all_names

    def get_magnets(self, data, all_names):
        df = self._pick_snapshot_df()
        magnet_mask = df.keys().str.endswith("I1")
        magnet_mask |= df.keys().str.endswith("I1D")

        new_data = df[magnet_mask].values
        new_names = list(df[magnet_mask].keys())

        data = np.append(data, new_data)
        all_names = np.append(all_names, new_names)

        return data, all_names

    def get_channels(self, data, all_names):
        df = self._pick_snapshot_df()
        set(df.keys()).intersection(self.snapshot.channels)
        # don't bother with columns that aren't in the original...
        new_names = list(set(df.keys()).intersection(self.snapshot.channels))
        new_data = df[new_names].values

        data = np.append(data, new_data)
        all_names = np.append(all_names, new_names)

        return data, all_names

    def get_single_image(self, data, all_names):
        ch = self.snapshot.images[0]
        df = self._pick_snapshot_df()
        path = Path(self.conf["data"]["basepath"]) / df[ch]
        path = path.with_suffix(".pcl")

        with path.open("rb") as f:
            img = pickle.load(f)
        data = np.append(data, df[ch])
        all_names = np.append(all_names, ch)

        image = TimestampedImage(ch, img, datetime.utcnow())
        return data, all_names, image

    def set_quad(self, name, value):
        channel = f"XFEL.MAGNETS/MAGNET.ML/{name}/KICK_MRAD.SP"
        LOG.info(f"Setting {channel} @ {value}")
        self.mi.set_value(channel, value)
        self.quads[name] = value
        # Then we must be starging a new measurement...
        self.bg_measurement_number = 0
        self.beam_measurement_number = 0

    def wait_machine_online(self):
        return  # Machine is always online if asked to be


class ScanType(Enum):
    DISPERSION = auto()
    TDS = auto()
    BETA = auto()

    @classmethod
    @property
    def ALT_NAME_MAP(cls):
        return {cls.DISPERSION: "dscan", cls.TDS: "tscan", cls.BETA: "bscan"}

    def get_filename(self, d, v):
        f"{}-{d=}-{v=}"


    def alt_name(self):
        return self.ALT_NAME_MAP[self]

    @classmethod
    def from_name(cls, scan_key):
        if scan_key == "tscan":
            return ScanType.TDS
        elif scan_key == "dscan":
            return ScanType.DISPERSION
        elif scan_key == "bscan":
            return ScanType.BETA


@dataclass
class SetpointMachineSnapshots:
    """Class for representing a set of machine snapshots at a single machine
    setpoint.  On top of the data read directly from the machine (the
    pd.DataFrame), there's also the scan_type, the dispersion setpoint
    and the corresponding measured dispersion.   TDS Amplitude is not
    recorded here because it is stored directly in the dataframe.

    """

    snapshots: pd.DataFrame
    scan_type: ScanType
    dispersion_setpoint: float
    measured_dispersion: Optional[tuple[float, float]]
    beta: float
    screen_channel: str

    def __len__(self) -> int:
        return len(self.snapshots)

    def timestamped_name(self) -> str:
        timestamp = time.strftime("%Y-%m-%d@%H:%M:%S")
        ampl_sp = _tds_amplitude_setpoint_from_df(self.snapshots)
        dispersion = self.dispersion_setpoint
        scan_name = self.scan_type.alt_name()
        return f"{timestamp}>>{scan_name}>>D={dispersion},TDS={ampl_sp}%.pcl"

    @property
    def tds_amplitude_setpoint(self) -> float:
        setpoints = self.snapshots[TDS_I1_AMPLITUDE_READBACK_ADDRESS]
        one_setpoint = setpoints.iloc[0]

        if not (setpoints == one_setpoint).all():
            raise ValueError("Setpoint is not consistent across snapshots")
        return one_setpoint

    def __repr__(self) -> str:
        tname = type(self).__name__
        dx0 = self.dispersion_setpoint
        try:
            dx1, dx1e = self.measured_dispersion
        except TypeError:
            dx1 = self.measured_dispersion
            dx1e = 0.0
        bstring = f", beta={self.beta}"

        try:
            sname = self.scan_type.name
        except AttributeError:
            sname = self.scan_type

        out = (
            f"<{tname}: scan_type={sname}, dxm=({dx1}±{dx1e})m,"
            f" nsnapshots={len(self)}{bstring}>"
        )
        return out

    def resolve_image_path(self, dirname: Path) -> None:
        paths = self.snapshots["XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"]
        self.snapshots["XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"] = dirname / paths

    def drop_bad_snapshots(self) -> None:
        df = self.snapshots
        df = df[df['XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ'].notna()]
        self.snapshots = df

    def drop_images(self, bad_image_names: list[str]) -> None:
        if not bad_image_names:
            return

        df = self.snapshots
        image_paths = df[self.screen_channel]
        # mask True for good images
        good_mask = np.ones_like(image_paths, dtype="bool")
        for image_name in bad_image_names:
            # Flipped to false for bad images...
            good_mask &= ~image_paths.str.contains(image_name)

        df = df[good_mask]
        dropped_images = image_paths[~good_mask]
        LOG.debug(f"Dropped bad_images: {dropped_images}")
        self.snapshots = df


def _tds_amplitude_setpoint_from_df(df):
    key_name = TDS.AMPLITUDE_RB
    col = df[key_name]
    value = col.iloc[0]
    if not (value == col).all():
        raise MalformedSnapshotDataFrame(
            f"{key_name} in {df} should be constant but is not"
        )
    return value


@dataclass
class MeasurementConfig:
    n_bg_images: int
    n_beam_images: int
    dscan: DispersionScanConfiguration
    tscan: TDSScanConfiguration


# here is the AUTOGAIN on/off address for OTRC.55.I1 camera (as an
# example) - "XFEL.DIAG/CAMERA/OTRC.58.I1/GAINAUTO.NUM". There are 3
# options: 0 - OFF, 1 - Once, 2 - Continuous. You may set it either to
# 1 before each of your measurement series or to 2 and then it would
# be adjusted more or less at each image. The readout address (also
# OTRC.55.I1 example) - "XFEL.DIAG/CAMERA/OTRC.58.I1/GAINRAW". I
# believe it is all you need. Also, for instance, mAtthias as I know
# uses the AUTOGAIN in his emmitance measurement tool and it works
# fine. Though I don't know which exactly Once or Continuous one.
