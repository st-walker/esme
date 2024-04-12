
# PYBIND11: /Users/xfeloper/.conda/envs/esme/bin/pybind11-config
# PKG_CONFIG: export PKG_CONFIG_PATH=/local/Darwin-x86_64/lib/pkgconfig


"""

EuXFEL machine interfaces

S.Tomin, 2017

"""

from __future__ import absolute_import, print_function

import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

import matplotlib
import numpy as np
import pandas as pd

from esme.control.exceptions import DOOCSReadError, DOOCSWriteError

import sys
sys.path.append("/Users/xfeloper/stwalker/llpydoocs/build")

try:
    import llpydoocs
except ImportError:
    pass


LOG = logging.getLogger(__name__)


def make_doocs_channel_string(facility="", device="", location="", prop=""):
    return f"{facility}/{device}/{location}/{prop}"



class DOOCSAddress:
    def __init__(self, facility="", device="", location="", prop=""):
        self.facility = facility
        self.device = device
        self.location = location
        self.prop = prop

    @classmethod
    def from_string(cls, string):
        components = string.split("/")
        if len(components) > 4:
            raise ValueError(f"Malformed DOOCs address string: {string}")
        return cls(*components)

    def resolve(self):
        return f"{self.facility}/{self.device}/{self.location}/{self.prop}"

    def filled(self, facility="", device="", location="", prop=""):
        facility = facility if facility else self.facility
        device = device if device else self.device
        location = location if location else self.location
        prop = prop if prop else self.prop
        address = f"{facility}/{device}/{location}/{prop}"
        return address

    def with_location(self, location):
        return f"{self.facility}/{self.device}/{location}/{self.prop}"

    def is_wildcard_address(self):
        return "*" in self.resolve()

    def get_wildcard_component(self):
        if not self.is_wildcard_address():
            raise ValueErro("Not a wildcard address")
        if "*" in self.facility:
            return self.facility
        elif "*" in self.device:
            return self.device
        elif "*" in self.location:
            return self.location
        elif "*" in self.prop:
            return self.prop

    def __str__(self):
        return self.resolve()

    def __repr__(self):
        return f'<{type(self).__name__} @ {hex(id(self))}: "{self.resolve()}">'


@dataclass
class TimestampedImage:
    channel: str
    image: np.array
    timestamp: datetime

    def name(self) -> str:
        cam_name = self.screen_name()
        return f"{cam_name}-{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    def screen_name(self) -> str:
        return Path(self.channel).parent.name


from abc import abstractmethod


class DOOCSInterfaceABC:
    @abstractmethod
    def get_value(self, channel: str) -> Any:
        pass

    @abstractmethod
    def set_value(self, channel: str, val: Any) -> None:
        pass

    @abstractmethod
    def get_charge(self) -> float:
        pass


class DOOCSInterface(DOOCSInterfaceABC):
    """
    Machine Interface for European XFEL
    """
    def __init__(self):
        # # Just fail immediately if there's no pydoocs...
        # try:
        pass
        # except ImportError as e:
        #     raise e
        # else:
        #     super().__init__(*args, **kwargs)

    def get_value(self, channel: str) -> Any:
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        LOG.debug("get_value: channel", channel)
        try:
            val = pydoocs.read(channel)
        except pydoocs.DoocsException as e:
            raise DOOCSReadError(channel) from e
        return val["data"]

    def set_value(self, channel: str, val: Any) -> None:
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        LOG.debug(f"pydoocs.write: {channel} -> {val}")
        try:
            pydoocs.write(channel, val)
        except pydoocs.DoocsException as e:
            raise DOOCSWriteError(channel, val) from e

    def get_charge(self) -> float:
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")

class Snapshot:
    def __init__(self, sase_sections=None):
        self.sase_sections = []
        self.orbit_sections = {}
        self.magnet_sections = {}
        self.phase_shifter_sections = {}
        self.undulators = {}
        self.channels = []
        self.channels_tol = []

        # alarm channels
        self.alarms = []

        # multidim data
        self.images = []
        self.image_folders = []

    def add_image(self, ch, folder):
        # check if folder exists and create if not
        if not os.path.exists(folder):
            os.makedirs(folder)
        if ch in self.images:
            print(f"WARNING: image channel is already added: {ch}")
            return
        self.images.append(ch)
        self.image_folders.append(folder)

    def add_orbit_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.orbit_sections:
            print(f"WARNING: channel is already added: {sec_id}")
            return
        self.orbit_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_magnet_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.magnet_sections:
            print(f"WARNING: channel is already added: {sec_id}")
            return
        self.magnet_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_undulator(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            print(f"WARNING: channel is already added: {sec_id}")
            return
        if sec_id in self.sase_sections:
            self.undulators[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_channel(self, channel, tol=None):
        if channel in self.channels:
            print(f"WARNING: channel is already added: {channel}")
            return
        self.channels.append(channel)
        self.channels_tol.append(tol)


class Machine:
    def __init__(
        self, snapshot: Snapshot, di: Optional[Type[DOOCSInterfaceABC]] = None
    ):
        self.snapshot = snapshot
        if di is None:
            di = DOOCSInterface()
        self.di = di
        self.bpm_server = "ORBIT"  # or "BPM"
        self.server = "XFEL"
        self.subtrain = "ALL"
        self.suffix = ""

    def is_machine_online(self) -> bool:
        """
        method to check if machine is online

        :return: True if online, False otherwise

        """

        for alarm in self.snapshot.alarms:
            # Read from the Machine the value
            val = self.di.get_value(alarm.channel)

            # Check if it's OK:
            if not alarm.is_ok(val):
                LOG.info(f"Machine is offline. Reason: {alarm.offline_message()}")
                return False

        return True

    def get_orbit(self, data, all_names):
        for sec_id in self.snapshot.orbit_sections:
            try:
                orbit_x = np.array(
                    self.di.get_value(
                        self.server
                        + ".DIAG/"
                        + self.bpm_server
                        + "/*."
                        + sec_id
                        + "/X."
                        + self.subtrain
                        + self.suffix
                    )
                )

                orbit_y = np.array(
                    self.di.get_value(
                        self.server
                        + ".DIAG/"
                        + self.bpm_server
                        + "/*."
                        + sec_id
                        + "/Y."
                        + self.subtrain
                        + self.suffix
                    )
                )
            except Exception as e:
                print("orbit id: " + sec_id + " ERROR: " + str(e))
                return [], []
            x = orbit_x[:, 1].astype(float)
            y = orbit_y[:, 1].astype(float)
            xy = np.append(x, y)

            names_x = [name + ".X" for name in orbit_x[:, 4]]
            names_y = [name + ".Y" for name in orbit_y[:, 4]]
            names = np.append(names_x, names_y)
            data = np.append(data, xy)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_magnets(self, data, all_names):
        for sec_id in self.snapshot.magnet_sections:
            try:
                magnets = np.array(
                    self.di.get_value(
                        "XFEL.MAGNETS/MAGNET.ML/*." + sec_id + "/KICK_MRAD.SP"
                    )
                )
            except Exception as e:
                print("magnets id: " + sec_id + " ERROR: " + str(e))
                return [], []
            vals = magnets[:, 1].astype(float)

            names = [name for name in magnets[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_channels(self, data, all_names):
        data = list(data)
        for ch in self.snapshot.channels:
            try:
                val = self.di.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                val = np.nan
            data.append(val)
            all_names = np.append(all_names, ch)
        return data, all_names

    def get_images(self, data, all_names):
        for i, ch in enumerate(self.snapshot.images):
            folder = self.snapshot.image_folders[i]
            try:
                img = self.di.get_value(ch)
            except Exception as e:
                print("id: " + ch + " ERROR: " + str(e))
                img = None

            cam_name = ch.split("/")[-2]
            # datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            # name = cam_name + "-" + time.strftime("%Y%m%d-%H%M%S")
            name = cam_name + "-" + datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]

            filename = name + ".png"
            path = str(folder) + os.sep + str(filename)
            path_pcl = str(folder) + os.sep + str(name) + ".pcl"
            if img is not None:
                # scipy.misc.imsave(path, img)
                # imageio.imwrite(path, img)
                matplotlib.image.imsave(path, img)
                with open(path_pcl, 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump(img, f)

            else:
                path = None
            # print(data)
            # data = list(data)
            # data = np.append(data, np.array([path], dtype=object))
            data.append(filename)
            all_names = np.append(all_names, ch)
        return data, all_names

    def get_single_image(self, data, all_names):
        ch = self.snapshot.images[0]
        image = TimestampedImage(ch, self.di.get_value(ch), datetime.utcnow())

        return data, all_names, image

    def wait_machine_online(self):
        if self.is_machine_online():
            return

        while True:
            if self.is_machine_online():
                print("machine is BACK. Wait 5 sec for recovering and continue")
                time.sleep(5)
                break
            else:
                time.sleep(3)
                print("machine is OFFLINE. Sleep 3 sec ..")

    def get_machine_snapshot(self, check_if_online=False):
        if check_if_online:
            self.wait_machine_online()

        data = np.array([time.time()], dtype=object)
        all_names = np.array(["timestamp"])
        data, all_names = self.get_orbit(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing orbit information, snapshot failed")
            return None

        # print(len(data), len(all_names))
        data_dict = {}
        for name, d in zip(all_names, data):
            data_dict[name] = [d]

        df = pd.DataFrame(data_dict, columns=data_dict.keys())
        static_bits, image = self.get_static_snapshot()
        return df.join(static_bits), image
        return pd.concat([df, static_bits]), image

    def get_static_snapshot(self):
        data = np.array([time.time()], dtype=object)
        all_names = np.array(["timestamp"])
        all_names = []
        data = []
        # data, all_names = self.get_orbit(data, all_names)

        data, all_names = self.get_magnets(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing magnet information, snapshot failed")
            return None
        data, all_names = self.get_channels(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing other channels, snapshot failed")
            return None
        data, all_names, image = self.get_single_image(data, all_names)
        all_names = np.append(all_names, image.channel)
        data.append("images/" + image.name() + ".png")
        if len(data) == 0:
            LOG.warning("Missing images, snapshot failed")
            return None

        data_dict = {}
        for name, d in zip(all_names, data):
            data_dict[name] = [d]

        df = pd.DataFrame(data_dict, columns=data_dict.keys())
        return df, image

    def get_wildcard(self, wildcard_string):
        # don't use arrays here so that we keep the original dataypes,
        # otherwise everything gets coerced to strings because the
        # last column is always a string (the address part).
        result = pydoocs.read(wildcard_string)["data"]
        names = [l[-1] for l in result]
        values = [[l[1]] for l in result]
        return dict(zip(names, values))


def is_in_controlroom():
    name = socket.gethostname()
    reg = re.compile(r"xfelbkr[0-9]\.desy\.de")
    return bool(reg.match(name))

def make_default_doocs_interface() -> DOOCSInterfaceABC:
    if not is_in_controlroom():
        return get_default_virtual_doocs_interface()
    else:
        return DOOCSInterface()
