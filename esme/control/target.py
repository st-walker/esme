from dataclasses import dataclass

from .taskomat import Sequence


@dataclass
class TargetDefinition:
    sequence: Sequence
    init_step: int
    property_to_dump: tuple[str, int]
    property_undo_dump: tuple[str, int]
    # Something something optics?
