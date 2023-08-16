from typing import Optional
import numpy.typing as npt

import logging
from dataclasses import dataclass

from esme.control.mint import XFELMachineInterface
from .kickers import FastKickerSetpoint
from .exceptions import EuXFELUserError
from .mint import XFELMachineInterface
from .sbunches import DiagnosticRegion

LOG = logging.getLogger(__name__)



class Screen:
    SCREEN_ML_FD = "XFEL.DIAG/SCREEN.ML/{}/{}"
    CAMERA_FD = "XFEL.DIAG/CAMERA/{}/{}"

    SCREEN_IS_POS_VALUE = 1
    CAMERA_IS_ON_VALUE = 1

    def __init__(self, name, location: DiagnosticRegion,
                 fast_kicker_setpoints: Optional[list[FastKickerSetpoint]] = None,
                 mi: Optional[XFELMachineInterface] = None) -> None:
        self.name = name
        self.location = location
        self.fast_kicker_setpoints = fast_kicker_setpoints
        self.mi = mi if mi else XFELMachineInterface()

    def is_offaxis(self):
        value = self.mi.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "OFFAXIS_LYSO"

    def is_onaxis(self):
        value = self.mi.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "ONAXIS_LYSO"

    def is_camera_on(self):
        value = self.mi.get_value(self.CAMERA_DF.format(self.name, "START"))
        return value == self.CAMERA_IS_ON_VALUE

    def is_screen_ok(self):
        return (self.is_in() or self.is_off_axis()) and self.is_on()
    

class ScreenService:
    SCREEN_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT"
    def __init__(self, screens: list[Screen], location: DiagnosticRegion = None, mi: Optional[XFELMachineInterface] = None) -> None:
        self.screens = screens
        self.location = location if location else DiagnosticRegion.UNKNOWN
        self.mi = mi if mi else XFELMachineInterface()

    def active_region_screen_names(self) -> list[str]:
        return [screen.name for screen in self.screens if screen.location == self.location]

    @property
    def screen_names(self) -> list[str]:
        return [screen.name for screen in self.screens]

    def get_screen(self, screen_name: str) -> Screen:
        index = self.screen_names.index(screen_name)
        return self.screens[index]
        
    def get_image(self, screen_name: str) -> npt.ArrayLike:
        ch = self.SCREEN_FDP_TEMPLATE.format(screen_name)
        LOG.info(f"Getting image from channel: {ch}")
        return self.mi.get_value(ch)
    
    def get_fast_kicker_setpoints_for_screen(self, screen_name: str) -> list[FastKickerSetpoint]:
        LOG.debug(f"Trying to get FastKickerSetpoint for screen: {screen_name}")
        screen = self.get_screen(screen_name)
        fast_kicker_setpoints = screen.fast_kicker_setpoints
        if fast_kicker_setpoints is None:
            raise EuXFELUserError("Screen has no fast kicker setpoint information")
        LOG.debug(f"Got FastKickerSetpoint for screen {screen_name}: {fast_kicker_setpoints}")
        return fast_kicker_setpoints
