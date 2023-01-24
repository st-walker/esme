"""

EuXFEL machine interfaces

S.Tomin, 2017

"""

from __future__ import absolute_import, print_function

import base64
import logging
import subprocess
import pickle
import os
import time
from datetime import datetime

import matplotlib
import pandas as pd
import numpy as np
try:
    import pydoocs
except ImportError:
    pass


LOG = logging.getLogger(__name__)


class EuXFELMachineError(RuntimeError):
    pass

class Device(object):
    def __init__(self, eid=None):
        self.eid = eid
        self.id = eid
        self.values = []
        self.times = []
        self.simplex_step = 0
        self.mi = None
        self.tol = 0.001
        self.timeout = 5  # seconds
        self.target = None
        self.low_limit = 0
        self.high_limit = 0
        self.phys2hw_factor = 1.0
        self.hw2phys_factor = 1.0

    def set_value(self, val):
        self.values.append(val)
        self.times.append(time.time())
        self.target = val
        self.mi.set_value(self.eid, val)

    def set_low_limit(self, val):
        self.low_limit = val

    def set_high_limit(self, val):
        self.high_limit = val

    def get_value(self):
        val = self.mi.get_value(self.eid)
        return val

    def trigger(self):
        pass

    def wait(self):
        if self.target is None:
            return

        start_time = time.time()
        while start_time + self.timeout <= time.time():
            if abs(self.get_value() - self.target) < self.tol:
                return
            time.sleep(0.05)

    def state(self):
        """
        Check if device is readable

        :return: state, True if readable and False if not
        """
        state = True
        try:
            self.get_value()
        except:
            state = False
        return state

    def clean(self):
        self.values = []
        self.times = []

    def check_limits(self, value):
        limits = self.get_limits()
        if value < limits[0] or value > limits[1]:
            print('limits exceeded for ', self.id, " - ", value, limits[0], value, limits[1])
            return True
        return False

    def get_limits(self):
        return [self.low_limit, self.high_limit]

    def phys2hw(self, phys_val):
        """
        Method to translate physical units to hardware units, e.g. angle [rad] to current [A]

        :param phys_val: physical unit
        :return: hardware unit
        """
        hw_val = phys_val * self.phys2hw_factor
        return hw_val

    def hw2phys(self, hw_val):
        """
        Method to translate hardware units to physical units, e.g. current [A] to angle [rad]

        :param hw_val: hardware unit
        :return: physical unit
        """
        phys_val = hw_val * self.hw2phys_factor
        return phys_val



class XFELMachineInterface:
    """
    Machine Interface for European XFEL
    """

    def __init__(self, args=None):
        super(XFELMachineInterface, self).__init__(args=args)
        self.logbook = "xfellog"
        self.uncheck_upstream_bpms = True  # uncheck bpms upstream the first corrector
        self.allow_star_operation = True
        self.hide_section_selection = False
        self.hide_close_trajectory = False
        self.hide_xfel_specific = False
        self.hide_dispersion_tab = False
        self.twiss_periodic = False
        self.analyse_correction = True


    def get_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        LOG.debug(" get_value: channel" + channel)
        val = pydoocs.read(channel)
        # print(channel, "   TIMESTAMP:",  val["timestamp"])

        return val["data"]

    def set_value(self, channel, val):
        """
        Method to set value to a channel

        :param channel: (str) String of the devices name used in doocs
        :param val: value
        :return: None
        """
        # print("SETTING")
        pydoocs.write(channel, val)
        return

    def get_raw_value(self, channel):
        """
        Getter function for XFEL.

        :param channel: (str) String of the devices name used in doocs
        :return: Data from pydoocs.read(), variable data type depending on channel
        """
        LOG.debug(" get_raw_value: channel" + channel)
        val = pydoocs.read(channel)
        return val

    def get_charge(self):
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")

    def send_to_logbook(self, *args, **kwargs):
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
                ['/usr/bin/lp', '-o', 'raw', '-d', elog], stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
            # send printer job
            lpr.communicate(elogXMLString.encode('utf-8'))
        except:
            succeded = False
        return succeded
    
class Machine:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.mi = XFELMachineInterface()
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
            try:
                val = self.mi.get_value(alarm)
            except Exception as exc:
                msg = f"Couldn't read alarm channel: {alarm.channel}"
                LOG.error(msg, exc_info=True)
                raise EuXFELMachineError(msg) from exc

            if not alarm.is_ok(val):
                log.info(f"Machine is offline. Reason: {alarm.offline_message()}")
                return False
        
        return True

    def get_orbit(self, data, all_names):
        for sec_id in self.snapshot.orbit_sections:
            try:
                orbit_x = np.array(
                    self.mi.get_value(
                        self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/X." + self.subtrain + self.suffix
                    )
                )

                orbit_y = np.array(
                    self.mi.get_value(
                        self.server + ".DIAG/" + self.bpm_server + "/*." + sec_id + "/Y." + self.subtrain + self.suffix
                    )
                )
            except Exception as e:
                print("orbit id: " + sec_id + " ERROR: " + str(e))
                return [], []
            x = orbit_x[:, 1].astype(np.float)
            y = orbit_y[:, 1].astype(np.float)
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
                magnets = np.array(self.mi.get_value("XFEL.MAGNETS/MAGNET.ML/*." + sec_id + "/KICK_MRAD.SP"))
            except Exception as e:
                print("magnets id: " + sec_id + " ERROR: " + str(e))
                return [], []
            vals = magnets[:, 1].astype(np.float)

            names = [name for name in magnets[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_phase_shifters(self, data, all_names):
        for sec_id in self.snapshot.phase_shifter_sections:
            try:
                phase_shifters = np.array(
                    self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/BPS.*." + sec_id + "/GAP.OFFSET")
                )
            except Exception as e:
                print("ERROR: magnets: ", sec_id, e)
                return [], []
            vals = phase_shifters[:, 1].astype(np.float)

            names = [name for name in phase_shifters[:, 4]]
            data = np.append(data, vals)
            all_names = np.append(all_names, names)
        return data, all_names

    def get_undulators(self, data, all_names):
        for sec_id in self.snapshot.undulators:
            try:
                if sec_id in ["SA1", "SA2"]:
                    undulators = np.array(
                        self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/U40.*." + sec_id + "/GAP")
                    )
                else:
                    undulators = np.array(
                        self.mi.get_value("XFEL.FEL/WAVELENGTHCONTROL." + sec_id + "/U68.*." + sec_id + "/GAP")
                    )
            except Exception as e:
                print("ERROR: magnets: ", sec_id, e)
                return [], []
            vals = undulators[:, 1].astype(np.float)

            names = [name for name in undulators[:, 4]]
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
            path = folder + os.sep + filename
            path_pcl = folder + os.sep + name + ".pcl"
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
            data.append(path)
            all_names = np.append(all_names, ch)
        return data, all_names

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

        # if not self.is_machine_online():
        #    print("machine is not online. wait 3 sec ...")
        #    time.sleep(2)
        #    return None
        if check_if_online:
            self.wait_machine_online()

        data = np.array([time.time()], dtype=object)
        all_names = np.array(["timestamp"])
        data, all_names = self.get_orbit(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing orbit information, snapshot failed")
            return None
        data, all_names = self.get_magnets(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing magnet information, snapshot failed")
            return None
        data, all_names = self.get_phase_shifters(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing phase shift information, snapshot failed")
            return None
        data, all_names = self.get_undulators(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing undulator information, snapshot failed")
            return None
        data, all_names = self.get_channels(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing other channels, snapshot failed")
            return None
        data, all_names = self.get_images(data, all_names)
        if len(data) == 0:
            LOG.warning("Missing images, snapshot failed")
            return None
        # print(len(data), len(all_names))
        data_dict = {}
        for name, d in zip(all_names, data):
            data_dict[name] = [d]

        df = pd.DataFrame(data_dict, columns=data_dict.keys())
        return df


class MPS(Device):
    def __init__(self, eid=None, server="XFEL", subtrain="SA1"):
        super(MPS, self).__init__(eid=eid)
        self.mi = XFELMachineInterface()
        self.subtrain = subtrain
        self.server = server

    def beam_off(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 0)

    def beam_on(self):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED", 1)

    def num_bunches_requested(self, num_bunches=1):
        self.mi.set_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/NUM_BUNCHES_REQUESTED_1", num_bunches)

    def is_beam_on(self):
        val = self.mi.get_value(self.server + ".UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED")
        return val


class CavityA1(Device):
    def __init__(self, eid, server="XFEL", subtrain="SA1"):
        super(CavityA1, self).__init__(eid=eid)
        self.subtrain = subtrain
        self.server = server

    def set_value(self, val):
        ch = self.server + ".RF/LLRF.CONTROLLER/" + self.eid + "/SP.AMPL"
        self.mi.set_value(ch, val)
        LOG.debug("CavityA1, ch: " + ch + " V = " + str(val))

    def get_value(self):
        ch = self.server + ".RF/LLRF.CONTROLLER/" + self.eid + "/SP.AMPL"
        val = self.mi.get_value(ch)
        return val


class BasicAlarm:
    def __init__(self, channel, vmin=-np.inf, vmax=+np.inf, explanation=""):
        self.channel = channel
        self.vmin = vmin
        self.vmax = vmax
        explanation = ""

    def is_ok(self, value):
        return (value >= self.vmin) and (value < self.vmax)

    def offline_message(self) -> str:
        return (f"{self.channel} out of bounds, bounds = {(self.vmin, self.vmax)}."
                f" explanation:  {self.explanation}")


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
            print("WARNING: image channel is already added")
            return
        self.images.append(ch)
        self.image_folders.append(folder)

    def add_orbit_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.orbit_sections:
            print("WARNING: channel is already added")
            return
        self.orbit_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_magnet_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.magnet_sections:
            print("WARNING: channel is already added")
            return
        self.magnet_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_phase_shifter_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.phase_shifter_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.phase_shifter_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_undulator(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.undulators[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_channel(self, channel, tol=None):
        if channel in self.channels:
            print("WARNING: channel is already added")
            return
        self.channels.append(channel)
        self.channels_tol.append(tol)

    
