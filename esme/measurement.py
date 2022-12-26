# Priority:

# Take data automatically.
# Should be able to turn TDS on/off
# Should be able to recover from TDS turning off randomly.
# Should keep track of good images/bad images.  Ignore bad ones.
# Should also write metadata nicely.
# Should automatically change voltage for TDS scan.
# Should automatically change quads for dispersion scan.
# Turn beam on and off
# Should check TDS is on, check beam is on, check screen is out.
# Autogain?


# this is a hobby, let's be honest..  let's go home and use our brain
# on something harder and do this in the evenings.

# Then do, say, measure dispersion automatically
# Finally, say, calibrate the TDS automatically

# Needs to handle:
# Beam turning off.
# TDS turning off.
# Repeating a measurement.


# I don't think I need multiple threads.  Just check TDS is on with
# the use of the machine alarm thing.  Eventually it should turn
# itself back on automatically, but for now it will be like with the
# beam on thing, it will just wait for it to be switched on!


import logging
import time
from dataclasses import dataclass
from pathlib import Path

import toml
import pandas as pd

from esme.mint.machine import MPS, Machine
from esme.channels import SNAPSHOT_TEMPL


LOG = logging.getLogger(__name__)

I1_DUMP_SCREEN_ADDRESS: str = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"
SNAPSHOT_TEMPL.add_image(I1_DUMP_SCREEN_ADDRESS, folder="./tds_images")

class MeasurementPrompt:
    pass


class EnergySpreadMeasurement:
    pass


# class PhotoShoot:
#     def __init__(self, x):
#         pass



@dataclass
class QuadrupoleSetting:
    names: list
    strengths: list
    dispersion: float

    def __post_init__(self):
        if len(self.names) != len(self.strengths):
            raise ValueError("Length mismatch between names and strengths")


@dataclass
class InjectorScanOpticsConfiguration:
    reference_setting: QuadrupoleSetting
    scan_settings: list[QuadrupoleSetting]

    @classmethod
    def from_conf(cls, config_path):
        # dscan_quads = []
        conf = toml.load(config_path)
        inj = conf["injector"]

        ref = inj["reference_optics"]
        reference_setting = QuadrupoleSetting(ref["names"],
                                              ref["strengths"],
                                              ref["dispersion"])

        scan_names = inj["scan_names"]
        scan_settings = []
        dscan = inj["dscan"]
        for settings in dscan.values():
            scan_settings.append(QuadrupoleSetting(scan_names,
                                                   settings["strengths"],
                                                   settings["dispersion"],
                                                   use_for_tds_scan))
        return cls(reference_optics, scan_settings)

    def quads_for_dispersion(self, dispersion):
        return next(x for x in self.scan_settings if x.dispersion == dispersion)


class MeasurementRunner():
    def __init__(self, basename, qsettings):
        self.basename = Path(basename)
        self.qsettings = qsettings

    def run(self, measure_dispersion):
        tds_scan()
        quad_scan()


    def set_reference_quads(self):
        pass

    def set_quads(self, dispersion, measure=True):
        LOG.info(f"Setting quads for dispersion @ {dispersion}")
        pass

    def set_tds_amplitude(self, amplitude):
        pass

    def dispersion_scan(self, tds_amplitude):
        self.set_tds_amplitude(tds_amplitude)
        # for

    def tds_scan(self, dispersion):
        self.set_quads(dispersion, measure=True)


class ScreenPhotographer:
    def __init__(self, mps=None, machine=None):
        self.mps = mps if mps is not None else MPS()
        self.machine = machine if machine is not None else Machine(SNAPSHOT_TEMPL)

    def take_background(self, nshots, delay=1):
        LOG.info("Taking background")
        self.switch_beam_off()
        time.sleep(delay)
        snapshots = []
        for i in range(nshots):
            LOG.info(f"Background image: {i} / {nshots - 1}")
            snapshots.append(self.take_machine_snapshot())
            time.sleep(delay)
        LOG.info("Finished taking background")
        self.switch_beam_on()
        return pd.DataFrame(snapshots)

    def take_data(self, nshots, delay=1):
        LOG.info("Taking data")
        self.switch_beam_on()
        snapshots = []

        mshots = 0
        while nshots < mshots:
            LOG.info(f"taking image {i}")
            snapshots.append(self.take_machine_snapshot())
            if self.is_machine_on():
                self.switch_beam_on()

            if 
            try:
                

            time.sleep(delay)
        return pd.DataFrame(snapshots)

    def take_machine_snapshot(self):
        LOG.info("Taking machine snapshot")
        return self.machine.get_machine_snapshot(check_if_online=True)

    def switch_beam_off(self):
        LOG.info("Switching beam off")
        self.mps.beam_off()

    def switch_beam_on(self):
        LOG.info("Switching beam ON")
        self.mps.beam_on()

    def is_machine_off(self):
        return not self.machine.is_machine_online()


def handle_sigint():
    pass


def save_snapshot(db, fname, metadata):
    LOG.info("Saving snapshot to df: {fname} with metadata: {metadata}")
    db.save(filename=fname)


def pop_df_row(df, row_index):
    df = df.reset_index(drop=True)
    index = df.index[row_index]
    result = df.iloc[index]
    df = df.drop(index=index)
    df = df.reset_index(drop=True)
    return df, result

def delete_image():
    pass
