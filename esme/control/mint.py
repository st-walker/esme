"""

EuXFEL machine interfaces

S.Tomin, 2017

"""

from __future__ import absolute_import, print_function

import base64
import logging
import os
import pickle
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type

import matplotlib
import numpy as np
import pandas as pd

try:
    import pydoocs
except ImportError:
    pass


LOG = logging.getLogger(__name__)


class EuXFELMachineError(RuntimeError):
    pass


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

    def filled(self, facility=None, device=None, location=None, prop=None):
        pass

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


class XFELMachineInterfaceABC:
    @abstractmethod
    def get_value(self, channel: str) -> Any:
        pass

    @abstractmethod
    def set_value(self, channel: str, val: Any) -> None:
        pass

    @abstractmethod
    def get_charge(self) -> float:
        pass


class XFELMachineInterface(XFELMachineInterfaceABC):
    """
    Machine Interface for European XFEL
    """
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
            raise pydoocs.DoocsException(f"Failed get_value with channel: {channel}") from e
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
            raise pydoocs.DoocsException(f"Failed writing {val} to {channel}") from e

    def get_charge(self) -> float:
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")

    def send_to_logbook(self, *args, **kwargs) -> bool:
        """
        Send information to the electronic logbook.

        :param args:
            Values sent to the method without keywork
        :param kwargs:
            Dictionary with key value pairs representing all the metadata
            that is available for the entry.
        :return: bool
            True when the entry was successfully generated, False otherwise.
        """
        author = kwargs.get('author', '')
        title = kwargs.get('title', '')
        severity = kwargs.get('severity', '')
        text = kwargs.get('text', '')
        image = kwargs.get('image', None)
        elog = self.logbook

        # The DOOCS elog expects an XML string in a particular format. This string
        # is beeing generated in the following as an initial list of strings.
        succeded = True  # indicator for a completely successful job
        # list beginning
        elogXMLStringList = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<entry>']

        # author information
        elogXMLStringList.append('<author>')
        elogXMLStringList.append(author)
        elogXMLStringList.append('</author>')
        # title information
        elogXMLStringList.append('<title>')
        elogXMLStringList.append(title)
        elogXMLStringList.append('</title>')
        # severity information
        elogXMLStringList.append('<severity>')
        elogXMLStringList.append(severity)
        elogXMLStringList.append('</severity>')
        # text information
        elogXMLStringList.append('<text>')
        elogXMLStringList.append(text)
        elogXMLStringList.append('</text>')
        # image information
        if image:
            try:
                encodedImage = base64.b64encode(image)
                elogXMLStringList.append('<image>')
                elogXMLStringList.append(encodedImage.decode())
                elogXMLStringList.append('</image>')
            except:  # make elog entry anyway, but return error (succeded = False)
                succeded = False
        # list end
        elogXMLStringList.append('</entry>')
        # join list to the final string
        elogXMLString = '\n'.join(elogXMLStringList)
        # open printer process
        try:
            lpr = subprocess.Popen(
                ['/usr/bin/lp', '-o', 'raw', '-d', elog],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
            # send printer job
            lpr.communicate(elogXMLString.encode('utf-8'))
        except:
            succeded = False
        return succeded


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
        self, snapshot: Snapshot, mi: Optional[Type[XFELMachineInterfaceABC]] = None
    ):
        self.snapshot = snapshot
        if mi is None:
            mi = XFELMachineInterface()
        self.mi = mi
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
            val = self.mi.get_value(alarm.channel)

            # Check if it's OK:
            if not alarm.is_ok(val):
                LOG.info(f"Machine is offline. Reason: {alarm.offline_message()}")
                return False

        return True

    def get_orbit(self, data, all_names):
        for sec_id in self.snapshot.orbit_sections:
            try:
                orbit_x = np.array(
                    self.mi.get_value(
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
                    self.mi.get_value(
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
                    self.mi.get_value(
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
                val = self.mi.get_value(ch)
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
                img = self.mi.get_value(ch)
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
        image = TimestampedImage(ch, self.mi.get_value(ch), datetime.utcnow())

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
        # from IPython import embed; embed()
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


class DictionaryXFELMachineInterface(XFELMachineInterfaceABC):
    def __init__(self, initial_state: Optional[dict] = None):
        self._machine_state = {}
        if initial_state is not None:
            self._machine_state |= initial_state

    def get_value(self, channel: str) -> Any:
        return self._machine_state[channel]

    def set_value(self, channel: str, val: Any) -> None:
        self._machine_state[channel] = val

    def get_charge(self) -> float:
        return 250e-12  # ?
    
