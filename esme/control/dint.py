# PYBIND11: /Users/xfeloper/.conda/envs/esme/bin/pybind11-config
# PKG_CONFIG: export PKG_CONFIG_PATH=/local/Darwin-x86_64/lib/pkgconfig


"""

EuXFEL machine interfaces

S.Tomin, 2017

"""

from __future__ import annotations

# Import hard coded pydoocs I compiled for Python 3.12
import importlib.util
import logging
import re
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self, Type

import numpy.typing as npt

from esme.control.exceptions import DOOCSReadError, DOOCSWriteError

PYDOOCS_SO = "/System/Volumes/Data/home/xfeloper/user/stwalker/pydoocs12/pydoocs-main/pydoocs.cpython-312-darwin.so"
SPEC = importlib.util.spec_from_file_location("pydoocs", PYDOOCS_SO)
if SPEC is not None:
    try:
        pydoocs = importlib.util.module_from_spec(SPEC)
    except ImportError:
        import warnings

        warnings.warn(f"Unable to import pydoocs from {PYDOOCS_SO}")


LOG = logging.getLogger(__name__)

DOOCS_RE = re.compile(
    r"^/?(?:[A-Z0-9._]+/)*[A-Z0-9._]*\*?[A-Z0-9._]*(?:/[A-Z0-9._]+)*/*$"
)

# This matches a full DOOCS address, with 4 fields, each field can
# contain caps, numbers, full stops and underscores.
DOOCS_FULL_ADDRESS_REGEX = re.compile(
    r"""
    ^                          # Start of string
    # /?                       # Possible leading slash

    # Facility:
    (?P<facility>              # Capture group for facility
    [0-9A-Z._]*                # Facility name, caps, numbers, full stops and underscores
    \*?                        # Optional wildcard character, at most once within the facility name
    [0-9A-Z._]*                # After the wildcard, more facility name characters
    )                          # End of facility capture group

    /                          # Separator between facility and device

    # Device:
    (?P<device>                # Capture group for device
    [0-9A-Z._]*                # Device name, caps, numbers, full stops and underscores
    \*?                        # Optional wildcard character, at most once within the device name
    [0-9A-Z._]*                # After the wildcard, more device name characters
    )                          # End of device capture group

    /                          # Separator between device and location

    # Location:
    (?P<location>              # Capture group for location.  Unlike other fields, can have lower
                               # case characters (I added this for the Taskomat sequences)
    [0-9A-Za-z._]*             # Location name, caps, numbers, full stops and underscores
    \*?                        # Optional wildcard character, at most once within the location name
    [0-9A-Za-z._]*             # After the wildcard, more location name characters
    )                          # End of location capture group

    /                          # Separator between location and property

    # Property:
    (?P<property>              # Capture group for property
    [0-9A-Z._]*                # Property name, caps, numbers, full stops and underscores
    \*?                        # Optional wildcard character, at most once within the property name
    [0-9A-Z._]*                # After the wildcard, more property name characters
    )                          # End of property capture group

    /?                         # Optional trailing slash
    $                          # End of string

""",
    re.VERBOSE,
)


def make_doocs_channel_string(
    facility: str = "", device: str = "", location: str = "", property: str = ""
) -> str:
    return f"{facility}/{device}/{location}/{property}"


class DOOCSAddress:
    __slots__ = "facility", "device", "location", "property"

    def __init__(
        self,
        facility: str = "",
        device: str = "",
        location: str = "",
        property: str = "",
    ):
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
        m = DOOCS_FULL_ADDRESS_REGEX.match(string)
        if m is None:
            raise ValueError(f"Malformed DOOCs address string: {string}")
        return cls(**m.groupdict())

    def filled(
        self,
        facility: str = "",
        device: str = "",
        location: str = "",
        property: str = "",
    ) -> str:
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
        return "*" in str(self)

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
        return f"{self.facility}/{self.device}/{self.location}/{self.property}"

    def __repr__(self) -> str:
        return f'<{type(self).__name__} @ {hex(id(self))}: "{str(self)}">'


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

    def get_names(self, wc_address: str) -> list[str]:
        try:
            sequence = pydoocs.names(wc_address)
        except pydoocs.DoocsException as e:
            raise DOOCSReadError(wc_address) from e

        try:
            return [s.split()[0] for s in sequence]
        except (IndexError, TypeError):
            raise DOOCSReadError(f"Malformed names output for {wc_address}") from e

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
        try:
            pydoocs.write(channel, val)
        except pydoocs.DoocsException as e:
            LOG.warning("Unable to write %s to write to %s", val, channel)
            raise DOOCSWriteError(channel, val) from e

    def get_charge(self) -> float:
        return self.get_value("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.SA1")


class VXFELDOOCSInterface(DOOCSInterface):
    def filter_address(self, address: str) -> str:
        rexp = re.compile("^/?XFEL")
        if not rexp.match(address):
            raise ValueError("Unable to redirect not XFEL address to Virtual XFEL")
        elif address.startswith("XFEL_SIM"):
            raise ValueError(f"{address} is already a virtual xfel address")

        return re.sub(rexp, "XFEL_SIM", address)

    def get_value(self, channel: str) -> Any:
        return super().get_value(self.filter_address(channel))
    
    def set_value(self, channel: str, value: Any) -> None:
        super().set_value(self.filter_address(channel), value)

    def get_names(self, wc_address: str) -> list[str]:
        return super().get_names(self.filter_address(wc_address))
    
    def read_full(self, channel: str) -> dict[str, Any]:
        return super().read_full(self.filter_address(channel))

def dump_fdl(
    stub: str,
    skip_types: set[str] | None = None,
    filter_regexes: list[str] | None = None,
) -> dict[str, Any]:
    skip_types = skip_types or {
        "TEXT",
        "SPECTRUM",
        "GSPECTRUM",
        "XML",
        "A_FLOAT",
        "A_INT",
        "IMAGE",
    }
    filter_regexes = filter_regexes or []
    if filter_regexes:
        pass
        # Compile into one mega regex..

    # Stub = something usable in pydoocs.names, e.g.
    # XFEL.DIAG/IMAGEANALYSIS/OTRC.58.I1/*/
    # The asterisk part HAS to be the property.
    addy = DOOCSAddress.from_string(stub)
    if "*" not in addy.property:
        raise ValueError("Malformed stub, property must be a wildcard")

    formattable_address = stub.replace("*", "{}")
    addresses = []
    values = []
    names = pydoocs.names(stub)
    if names[0] == "NAME = location":
        names = names[1:]

    for name in names:
        address_part, *_ = name.split()
        if address_part.endswith("DESC") and skip_disc:
            continue
        # if address_part.endswitih("CATEG") and skip_categ
        address = formattable_address.format(address_part)

        try:
            ddata = pydoocs.read(address)
        except pydoocs.DoocsException:
            continue

        dtype = ddata["type"]
        if ddata["type"] in skip_types:
            continue
        # Always skip images...
        if dtype == "image":
            continue

        addresses.append(address)
        values.append(ddata["data"])

    return dict(zip(addresses, values))
