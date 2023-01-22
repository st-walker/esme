"""
XFEL machine interface
S.Tomin, 2017
"""
from __future__ import absolute_import, print_function

try:
    import pydoocs
except ImportError:
    pass

import base64
import logging
import subprocess

from . import devices
from .interface import Device, MachineInterface

LOG = logging.getLogger(__name__)
# from ocelot.cpbd.response_matrix import *


class AlarmDevice(Device):
    """
    Devices for getting information about Machine status
    """

    def __init__(self, eid=None):
        super(AlarmDevice, self).__init__(eid=eid)


class XFELMachineInterface(MachineInterface):
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
        # self.orm_method = LinacRmatrixRM
        # self.drm_method = LinacDisperseSimRM
        # self.lattice_manager = lattice_manager
        self.devices = devices

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
