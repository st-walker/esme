
# PYBIND11: /Users/xfeloper/.conda/envs/esme/bin/pybind11-config
# PKG_CONFIG: export PKG_CONFIG_PATH=/local/Darwin-x86_64/lib/pkgconfig


"""

EuXFEL machine interfaces

S.Tomin, 2017

"""

from __future__ import annotations

import logging
import os
import pickle
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Type, Self
from abc import abstractmethod

import matplotlib
import numpy as np
import numpy.typing as npt
import pandas as pd

from esme.control.exceptions import DOOCSReadError, DOOCSWriteError

import sys

# Import hard coded pydoocs I compiled for Python 3.12
import importlib.util

PYDOOCS_SO = "/System/Volumes/Data/home/xfeloper/user/stwalker/pydoocs12/pydoocs-main/pydoocs.cpython-312-darwin.so"
SPEC = importlib.util.spec_from_file_location("pydoocs", PYDOOCS_SO)
if SPEC is not None:
    try:
        pydoocs = importlib.util.module_from_spec(SPEC)
    except ImportError:
        import warnings
        warnings.warn(f"Unable to import pydoocs from {PYDOOCS_SO}")


LOG = logging.getLogger(__name__)


def make_doocs_channel_string(facility: str = "", device: str = "", location: str = "", property: str = "") -> str:
    return f"{facility}/{device}/{location}/{property}"


class DOOCSAddress:
    __slots__ = "facility", "device", "location", "property"
    def __init__(self, facility: str = "", device: str = "", location: str = "", property: str = ""):
        self.facility = facility
        self.device = device
        self.location = location
        self.property = property
        if "/" in facility:
            raise ValueError("/ found in facility component")
        if "/" in device:
            raise ValueError("/ found in device component")
        if "/" in location:
            raise ValueError("/ found in location component")
        if "/" in property:
            raise ValueError("/ found in property component")

    @classmethod
    def from_string(cls: Type[Self], string: str) -> Self:
        components = string.split("/")
        if len(components) > 4:
            raise ValueError(f"Malformed DOOCs address string: {string}")
        return cls(*components)

    def resolve(self) -> str:
        return f"{self.facility}/{self.device}/{self.location}/{self.property}"

    def filled(self, facility: str = "", device: str = "", location: str = "", property: str= "") -> str:
        facility = facility or self.facility
        device = device or self.device
        location = location or self.location
        property = property or self.property
        address = f"{facility}/{device}/{location}/{property}"
        return address

    def filled_wildcard(self, substrings: list[str]) -> list[str]:
        if not self.is_wildcard_address():
            raise ValueError("Not a wildcard address")
        result = []
        facility = self.facility
        device = self.device
        location = self.location
        property = self.property
        if "*" in self.facility:
            for ss in substrings:
                result.append(f"{ss}/{self.device}/{self.location}/{self.property}")
        elif "*" in self.device:
            for ss in substrings:
                result.append(f"{self.facility}/{ss}/{self.location}/{self.property}")
        elif "*" in self.location:
            for ss in substrings:
                result.append(f"{self.facility}/{self.device}/{ss}/{self.property}")
        else:
            for ss in substrings:
                result.append(f"{self.facility}/{self.device}/{self.location}/{ss}")
        return result

    def with_location(self, location: str) -> str:
        return f"{self.facility}/{self.device}/{location}/{self.property}"

    def is_wildcard_address(self) -> bool:
        return "*" in self.resolve()

    def get_wildcard_component(self) -> str:
        if not self.is_wildcard_address():
            raise ValueError("Not a wildcard address")
        if "*" in self.facility:
            return self.facility
        elif "*" in self.device:
            return self.device
        elif "*" in self.location:
            return self.location
        elif "*" in self.property:
            return self.property
        raise ValueError("Unable to find wildcard component")

    def __str__(self) -> str:
        return self.resolve()

    def __repr__(self) -> str:
        return f'<{type(self).__name__} @ {hex(id(self))}: "{self.resolve()}">'


@dataclass
class TimestampedImage:
    channel: str
    image: npt.ArrayLike
    timestamp: datetime

    def name(self) -> str:
        cam_name = self.screen_name()
        return f"{cam_name}-{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')[:-3]}"

    def screen_name(self) -> str:
        return Path(self.channel).parent.name


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
        try:
            val = pydoocs.read(channel)
        except pydoocs.DoocsException as e:
            raise DOOCSReadError(channel) from e
        return val["data"]
    
    def read_full(self, channel: str) -> dict[str, Any]:
        try:
            return pydoocs.read(channel)
        except pydoocs.DocosException as e:
            raise DOOCSReadError(channel) from e
        
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
