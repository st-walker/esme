import re
from enum import Enum


class DiagnosticRegion(str, Enum):
    I1 = "I1"
    B1 = "B1"
    B2 = "B2"


def region_from_screen_name(screen_name: str) -> DiagnosticRegion:
    rmatch = re.match(".*([IB][12])D?$", screen_name)
    if rmatch is None:
        raise ValueError(f"Unable to extract region from screen name: {screen_name}")
    rstring, *_ = rmatch.groups()
    return DiagnosticRegion(rstring)
