from typing import Optional
import numpy.typing as npt

import logging
from dataclasses import dataclass

from .dint import DOOCSInterface
from .kickers import FastKickerSetpoint
from .exceptions import EuXFELUserError
from .dint import DOOCSInterface
from esme.core import DiagnosticRegion

LOG = logging.getLogger(__name__)


# @dataclass
# class ImageDimension:
    

class Screen:
    SCREEN_ML_FD = "XFEL.DIAG/SCREEN.ML/{}/{}"
    CAMERA_FD = "XFEL.DIAG/CAMERA/{}/{}"

    SCREEN_IS_POS_VALUE = 1
    CAMERA_IS_ON_VALUE = 1

    SCREEN_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT"
    SCREEN_RAW_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT_ZMQ"    

    
    def __init__(self, name,
                 fast_kicker_setpoints: Optional[list[FastKickerSetpoint]] = None,
                 di: Optional[DOOCSInterface] = None) -> None:
        self.name = name
        self.fast_kicker_setpoints = fast_kicker_setpoints
        self.di = di if di else DOOCSInterface()
        
    def is_offaxis(self):
        value = self.di.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "OFFAXIS_LYSO"

    def is_onaxis(self):
        value = self.di.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "ONAXIS_LYSO"

    def is_camera_on(self):
        value = self.di.get_value(self.CAMERA_DF.format(self.name, "START"))
        return value == self.CAMERA_IS_ON_VALUE

    def is_screen_ok(self):
        return (self.is_in() or self.is_off_axis()) and self.is_on()

    def get_pixel_xsize(self) -> float:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/X.POLY_SCALE" # mm
        return abs(self.di.get_value(addy)[2] * 1e-3) # mm to m

    def get_pixel_ysize(self) -> float:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/Y.POLY_SCALE" # mm
        return abs(self.di.get_value(addy)[2] * 1e-3) # mm to m

    def get_image_xpixels(self) -> int:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/WIDTH"
        return self.di.get_value(addy)

    def get_image_ypixels(self) -> int:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/HEIGHT"
        return self.di.get_value(addy)

    def get_image_width(self) -> float:
        return self.get_image_xpixels() * self.get_pixel_xsize()

    def get_image_height(self) -> float:
        return self.get_image_ypixels() * self.get_pixel_ysize()
        
    def get_image(self) -> npt.ArrayLike:
        ch = self.SCREEN_FDP_TEMPLATE.format(self.name)
        LOG.debug(f"Getting image from channel: {ch}")
        return self.di.get_value(ch)
    
    def get_image_raw(self) -> npt.ArrayLike:
        ch = self.get_image_raw_address(self.name)
        LOG.debug(f"Getting raw image from channel: {ch}")
        return self.di.get_value(ch)

    def get_image_raw_address(self,) -> npt.ArrayLike:
        return self.SCREEN_RAW_FDP_TEMPLATE.format(self.name)
    
    def get_fast_kicker_setpoints(self) -> list[FastKickerSetpoint]:
        LOG.debug(f"Trying to get FastKickerSetpoint for screen: {self.name}")
        fast_kicker_setpoints = self.fast_kicker_setpoints
        if fast_kicker_setpoints is None:
            raise EuXFELUserError("Screen has no fast kicker setpoint information")
        LOG.debug(f"Got FastKickerSetpoint for screen {self.name}: {fast_kicker_setpoints}")
        return fast_kicker_setpoints

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"


