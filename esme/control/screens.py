from typing import Optional
import numpy.typing as npt

import logging

from .dint import DOOCSInterface
from .kickers import FastKickerSetpoint
from .exceptions import EuXFELUserError, DOOCSUnexpectedReadValueError, DOOCSReadError
from .dint import DOOCSInterface

LOG = logging.getLogger(__name__)


# @dataclass
# class ImageDimension:
    

class Screen:
    SCREEN_ML_FD = "XFEL.DIAG/SCREEN.ML/{}/{}"
    CAMERA_FD = "XFEL.DIAG/CAMERA/{}/{}"

    SCREEN_IS_POS_VALUE = 1
    CAMERA_IS_TAKING_DATA_VALUE = 1

    SCREEN_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT"
    SCREEN_RAW_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT_ZMQ"

    POWER_ON_OFF_TEMPLATE = "XFEL.DIAG/CAMERA/{}/POWER.ON.OFF"

    def __init__(self, name,
                 fast_kicker_setpoints: Optional[list[FastKickerSetpoint]] = None,
                 di: Optional[DOOCSInterface] = None) -> None:
        self.name = name
        self.fast_kicker_setpoints = fast_kicker_setpoints
        self.di = di if di else DOOCSInterface()
    
    def read_camera_status(self) -> str:
        return self.di.get_value(self.CAMERA_FD.format(self.name, "CAM.STATUS"))
    
    def is_online(self) -> bool:
        return self.read_camera_status() == "Online"

    def is_offaxis(self) -> bool:
        value = self.di.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "OFFAXIS_LYSO"

    def is_onaxis(self) -> bool:
        value = self.di.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        return value == "ONAXIS_LYSO"

    def is_camera_taking_data(self):
        value = self.di.get_value(self.CAMERA_FD.format(self.name, "START"))
        return value == self.CAMERA_IS_TAKING_DATA_VALUE

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
        ch = self.get_image_raw_address()
        LOG.debug(f"Getting raw image from channel: {ch}")
        return self.di.get_value(ch)

    def get_image_raw_address(self) -> npt.ArrayLike:
        return self.SCREEN_RAW_FDP_TEMPLATE.format(self.name)
    
    def get_fast_kicker_setpoints(self) -> list[FastKickerSetpoint]:
        LOG.debug(f"Trying to get FastKickerSetpoint for screen: {self.name}")
        fast_kicker_setpoints = self.fast_kicker_setpoints
        if fast_kicker_setpoints is None:
            raise EuXFELUserError("Screen has no fast kicker setpoint information")
        LOG.debug(f"Got FastKickerSetpoint for screen {self.name}: {fast_kicker_setpoints}")
        return fast_kicker_setpoints
    
    def _power_on_off(self, *, on: bool) -> None:
        self.di.set_value(self.POWER_ON_OFF_TEMPLATE.format(self.name), int(on))

    def power_on(self) -> None:
        self._switch_on_off(on=True)

    def power_off(self) -> None:
        self._switch_on_off(on=False)

    def is_powered(self) -> bool:
        # If address read is 1 then it's powered, if it's zero then it's off.
        address = f"XFEL.DIAG/OTR.MOTOR/DOUT.{self.name}/CCD"
        value = self.di.get_value(address)
        # This is very defensive, I could just call bool on the return value but it's probably better to be careful.
        if value == 1:
            return True
        elif value == 0:
            return False
        else:
            raise DOOCSUnexpectedReadValueError(address, value)
        
    def is_responding(self) -> bool:
        if not self.is_powered():
            return False
        # If we cannot read the camera's image width, then this means the camera is not responding.  This is not necessarily something bad,
        # for example if the camera is in the process of booting.
        try:
            self.get_image_width()
        except DOOCSReadError:
            return False
        return True

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"
