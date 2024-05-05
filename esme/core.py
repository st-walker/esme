import re
from enum import Enum


class DiagnosticRegion(str, Enum):
    I1 = "I1"
    B1 = "B1"
    B2 = "B2"


def region_from_screen_name(screen_name: str) -> DiagnosticRegion:
    try:
        rstring, = re.match(".*([IB][12])D?$", screen_name).groups()
    except TypeError:
        raise f"Unable to extract region from screen name: {screen_name}"
    else:
        return DiagnosticRegion(rstring)
