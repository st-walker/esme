"""Some things to watch out for when dealing with Screens.  Regarding
a camera's state, the most important thing is ultimately whether, when
we try to read from its doocs address (Screen.get_image below), we get
a meaningful image returned as we would usually like.

There are two main addresses we care about for this.
Using OTRC.55.I1 as an example:

1. Is the camera powered on?
   We use the address XFEL.DIAG/OTR.MOTOR/DOUT.OTRC.55.I1/CCD.
   Screen instance method: is_powered
2. Is the camera set to actually acquire images?
   We use the address XFEL.DIAG/CAMERA/OTRC.55.I1/START
   Screen instance method: is_acquiring_images

For a camera to be powered on (1) but not acquiring images (2), this
makes some sense.  But here it's actually possible also for the camera
to be "acquiring images" (2) but not actually powered on.  So if you just read
XFEL.DIAG/CAMERA/OTRC.55.I1/START and get the value 1 in return, it's possible
still that we do not have any imagies actually being acquired if the

in short if you want to check if your screen is ACTUALLY acquiring images, you need to do:

s = Screen("OTRC.55.I1") # for example
is_actually_really_taking_images = s.is_powered() and s.is_acquiring_images()

"""

import logging
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from typing import Any

import numpy.typing as npt

from .dint import DOOCSInterface
from .exceptions import DOOCSReadError, DOOCSUnexpectedReadValueError, EuXFELUserError
from .kickers import FastKickerSetpoint

LOG = logging.getLogger(__name__)


@dataclass
class ScreenMetadata:
    xsize: float
    ysize: float
    nx: int
    ny: int


class PoweringState(Enum):
    STATIC = auto()
    UP = auto()
    DOWN = auto()

class Position(Enum):
    ONAXIS = auto()
    OFFAXIS = auto()
    OUT = auto()
    UNKNOWN = auto()

    @classmethod
    def from_doocs():
        return self

_DOOCS_STRING_TO_POSITION = {"OFFAXIS_LYSO": Position.OFFAXIS,
                             "ONAXIS_LYSO": Position.ONAXIS,
                             "OUT": Position.OUT}


class Screen:
    SCREEN_ML_FD = "XFEL.DIAG/SCREEN.ML/{}/{}"
    CAMERA_FD = "XFEL.DIAG/CAMERA/{}/{}"

    SCREEN_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT"
    SCREEN_RAW_FDP_TEMPLATE = "XFEL.DIAG/CAMERA/{}/IMAGE_EXT_ZMQ"

    POWER_ON_OFF_TEMPLATE = "XFEL.DIAG/CAMERA/{}/POWER.ON.OFF"

    def __init__(
        self,
        name: str,
        fast_kicker_setpoints: list[FastKickerSetpoint] | None = None,
        di: DOOCSInterface | None = None,
    ) -> None:
        self.name = name
        self.fast_kicker_setpoints = fast_kicker_setpoints
        self.di = di if di else DOOCSInterface()

    def read_camera_status(self) -> str:
        # I don't use this generally because I find it not very
        # useful, the camera may indeed be online but it doesn't tell
        # you if the camera is actually taking data for example.
        return self.di.get_value(self.CAMERA_FD.format(self.name, "CAM.STATUS"))
    
    def get_position(self) -> Position:
        pos = self.di.get_value(self.SCREEN_ML_FD.format(self.name, "STATUS.STR"))
        try:
            return _DOOCS_STRING_TO_POSITION[pos]
        except KeyError:
            LOG.critical("Position of screen %s is unknown, pos = %s", self.name, pos)
            return Position.UNKNOWN
    
    def get_pixel_xsize(self) -> float:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/X.POLY_SCALE"  # mm
        return abs(self.di.get_value(addy)[2] * 1e-3)  # mm to m

    def get_pixel_ysize(self) -> float:
        addy = f"XFEL.DIAG/CAMERA/{self.name}/Y.POLY_SCALE"  # mm
        return abs(self.di.get_value(addy)[2] * 1e-3)  # mm to m

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

    def get_image_compressed(self) -> npt.NDArray:
        """Not necessarily compressed, depends on
        XFEL.DIAG/CAMERA/{camera_name}/COMP_MODE state

        """
        return self.di.get_value(self.SCREEN_FDP_TEMPLATE.format(self.name))

    def get_image_raw(self) -> npt.NDArray | None:
        # Sometimes getting on the address doesn't fail, but just returns None
        # I don't know why.  It only happened once.
        return self.di.get_value(self.get_image_raw_address())
    
    def get_image_raw_full(self) -> dict[str, Any]:
        return self.di.read_full(self.get_image_raw_address())

    def get_image_raw_address(self) -> str:
        return self.SCREEN_RAW_FDP_TEMPLATE.format(self.name)

    def get_fast_kicker_setpoints(self) -> list[FastKickerSetpoint]:
        LOG.debug(f"Trying to get FastKickerSetpoint for screen: {self.name}")
        fast_kicker_setpoints = self.fast_kicker_setpoints
        if fast_kicker_setpoints is None:
            raise EuXFELUserError("Screen has no fast kicker setpoint information")
        LOG.debug(
            f"Got FastKickerSetpoint for screen {self.name}: {fast_kicker_setpoints}"
        )
        return fast_kicker_setpoints

    def get_powering_state(self) -> PoweringState:
        """Returns whether the screen is powering up, powering down, or neither"""
        ch = self.POWER_ON_OFF_TEMPLATE.format(self.name)
        match v := self.di.get_value(ch):
            case 0:
                return PoweringState.STATIC
            case 1:
                return PoweringState.UP
            case 2:
                return PoweringState.DOWN
            case _:
                raise DOOCSUnexpectedReadValueError(ch, v)

    def is_powered(self) -> bool:
        # If address read is 1 then it's powered, if it's zero then it's off.
        address = f"XFEL.DIAG/OTR.MOTOR/DOUT.{self.name}/CCD"
        value = self.di.get_value(address)
        # This is very defensive, I could just call bool on the return
        # value but it's probably better to be careful, because I don't entirely
        # know what values are possible here.
        if value == 1:
            return True
        elif value == 0:
            return False
        else:
            raise DOOCSUnexpectedReadValueError(address, value)

    def power_on_off(self, *, on: bool) -> None:
        ch = self.POWER_ON_OFF_TEMPLATE.format(self.name)
        if on:
            self.di.set_value(ch, 1)  # Magic number = 1 for powering on
        else:
            self.di.set_value(ch, 2)  # Magic number = 2 for powering off

    def _image_acquisition_address(self) -> str:
        """The address for whether or not this camera is set to acquire images"""
        return self.CAMERA_FD.format(self.name, "START")

    def start_stop_image_acquisition(self, *, acquire: bool = True) -> None:
        self.di.set_value(self._image_acquisition_address(), int(acquire))

    def is_acquiring_images(self) -> bool:
        return bool(self.di.get_value(self._image_acquisition_address()))

    def __repr__(self) -> str:
        return f"<{type(self).__name__}: {self.name}>"

    @cache
    def get_screen_metadata(self) -> ScreenMetadata:
        try:
            xsize = self.get_pixel_xsize()
            ysize = self.get_pixel_ysize()
            nx = self.get_image_xpixels()
            ny = self.get_image_ypixels()
        except DOOCSReadError as e:
            raise DOOCSReadError(
                "Unable to read screen pixel info, screen may be off"
            ) from e
        pix = ScreenMetadata(xsize=xsize, ysize=ysize, nx=nx, ny=ny)
        return pix


def screen_is_fully_operational(screen: Screen) -> bool:
    return screen.is_powered() and screen.is_acquiring_images()


# def try_and_boot_screen(screen: Screen) -> None:
#     if not screen.is_powered():
#         screen.power_on_off(on=True)
#     else:
#         screen.start_stop_image_acquisition(acquire=True)
async def try_and_boot_screen(screen: Screen, ntries: int = 1) -> None:
    for _ in range(ntries):
        await _try_and_boot_screen()
        is_all_good = screen_is_fully_operational(screen)


async def _try_and_boot_screen(screen: Screen) -> None:
    """The idea behind this coroutine is to try to boot a screen from scratch.

    This consists of two steps.

    """

    pstate = screen.get_powering_state()
    if pstate is not PoweringState.STATIC:
        await asyncio.sleep(1.0)

    if not screen.is_powered():
        screen.power_on_off(on=True)
        await asyncio.sleep(1.0)
    elif not screen.is_acquiring_images():
        screen.start_stop_image_acquisition(acquire=True)
        await asyncio.sleep(1.0)
