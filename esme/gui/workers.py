from enum import Enum, auto
from dataclasses import dataclass, field
from functools import cached_property, cache
from collections import deque
import queue
import time
import logging
from typing import Any
from copy import deepcopy
import threading

from PyQt5.QtCore import pyqtSignal, QObject, QTimer
import numpy.typing as npt
import numpy as np
from scipy import ndimage

from esme.control.screens import Position, Screen, ScreenMetadata
from esme.maths import get_gaussian_fit
from esme.image import zero_off_axis_regions
from esme.control.exceptions import DOOCSReadError

LOG = logging.getLogger(__name__)


@dataclass
class GaussianParameters:
    scale: float
    mu: float
    sigma: float

    def to_popt(self) -> tuple[float, float, float]:
        return (self.scale, self.mu, self.sigma)

@dataclass
class ImagePayload:
    image: npt.NDArray
    screen_md: ScreenMetadata
    time_calibration: float
    bunch_charge: float | None
    is_bg: bool

    @cached_property
    def eproj(self):
        return self.image.sum(axis=1)
    
    @cached_property
    def tproj(self):
        return self.image.sum(axis=0)

    @property
    def timepx(self):
        # Time is always the x-axis in our convention
        sh = self.image.shape
        return np.linspace(-sh[1] / 2, sh[1] / 2, num=len(self.tproj))
    
    @property
    def energypx(self):
        sh = self.image.shape
        return np.linspace(-sh[0] / 2, sh[0] / 2, num=len(self.eproj))    

    @property
    def time(self) -> npt.NDArray:
        return self.timepx * self.screen_md.ysize
    
    @property
    def energy(self) -> npt.NDArray:
        # Time is always the x-axis so energy is y-axis.
        return self.energypx * self.screen_md.xsize
    
    @property
    def time_calibrated(self) -> npt.NDArray:
        return self.time / self.time_calibration    

    @cached_property
    def current(self):
        profile = self.tproj * (self.bunch_charge / np.trapz(self.tproj, self.time_calibrated))
        print(self.bunch_charge, profile.max())
        return profile
    
    @cached_property
    def stdev_bunch_length(self) -> float:
        tcal = self.time_calibrated
        pdf = self.tproj / np.trapz(self.tproj, tcal)
        mean = np.trapz(pdf * self.time_calibrated, tcal)
        stdev = np.sqrt(np.trapz(pdf * (mean - tcal)**2, tcal))
        return stdev
    
    @cached_property
    def gaussfit_time(self):
        # This is uncalibrated time...
        popt, _ = get_gaussian_fit(self.time, self.tproj, p0=[self.tproj.max(), self.time[self.tproj.argmax()], 300 * self.screen_md.ysize])
        scale, mu, sigma = popt
        return GaussianParameters(scale=scale, mu=mu, sigma=sigma)

    @cached_property
    def gaussfit_current(self):
        popt, _ = get_gaussian_fit(self.time_calibrated, 
                                   self.current,
                                   p0=[self.current.max(),
                                       self.time_calibrated[self.current.argmax()],
                                       300 * self.screen_md.ysize / self.time_calibration])
        scale, mu, sigma = popt
        return GaussianParameters(scale=scale, mu=mu, sigma=sigma)


class MessageType(Enum):
    CACHE_BACKGROUND = auto()
    CLEAR_BACKGROUND = auto()
    # TAKE_N_FAST = auto()
    CHANGE_SCREEN = auto()
    SUBTRACT_BACKGROUND = auto()
    PLAY_SCREEN = auto()
    PAUSE_SCREEN = auto()
    SET_FREQUENCY = auto()
    CLIP_OFFAXIS = auto()
    SMOOTH_IMAGE = auto()
    TIME_CALIBRATION = auto()


@dataclass
class ImagingMessage:
    mtype: MessageType
    data: dict[str, Any] = field(default_factory=dict)


class StopImageAcquisition(Exception):
    pass


@dataclass
class ImageQueue:
    q: queue.Queue[ImagePayload]
    num_images_remaining: int
        

class ImagingWorker(QObject):
    display_image_signal = pyqtSignal(ImagePayload)
    nbackground_taken = pyqtSignal(int)
    finished_background_acquisition = pyqtSignal()

    def __init__(self, initial_screen: Screen):
        super().__init__()
        # The background images cache.
        self.bg_images: deque[npt.NDArray] = deque(maxlen=5)
        self.mean_bg = 0.0

        self.screen: Screen = initial_screen
        # This clearly assumes the screen is already powered etc..  no catching here!
        print(self.screen)
        self.screen_md: ScreenMetadata = initial_screen.get_screen_metadata()
        self._set_screen(initial_screen)

        # Flag for when possibly emitting a processed image for
        # display: Whether we should subtract the background for the
        # displayed image or not.
        self._do_subtract_background: bool = False
        self._caching_background: bool = False
        self._pause: bool = False
        self._read_period: float = 1.0
        self._clip_offaxis = False
        self._xyflip = False, False
        self._smooth_image = False
        self._time_calibration = None
        self._bunch_charge = None

        # Queue for messages we we receive from
        self._consumerq: queue.Queue[ImagingMessage | None] = queue.Queue()
        self._subscriber_queues: list[ImageQueue] = []
        self._lock = threading.Lock()

        self._is_running = False

    def get_cached_background(self) -> deque[npt.NDArray]:
        return deepcopy(self.bg_images)
    
    def bg_cache_is_empty(self) -> bool:
        return len(self.bg_images) == 0

    def bg_cache_is_full(self) -> bool:
        return len(self.bg_images) == self.bg_images.maxlen

    def run(self) -> None:
        if self._is_running:
            raise RuntimeError(
                "Already running, run should not be called more than once."
            )
        self._is_running = True
        self._slow_loop()

    def stop_read_loop(self):
        self._consumerq.put(None)

    def _slow_loop(self):
        """This is a slow loop that is the one that is running by default and acquires images at a rate of
        1Hz.
        """
        deadline = time.time() + self._read_period
        while True:
            # Only do this event loop about once a second
            wait = deadline - time.time()
            if wait > 0:
                time.sleep(wait)

            # set deadline for next read.
            deadline += self._read_period

            # Check for anything in the message queue
            try:
                self._dispatch_from_message_queue()
            except StopImageAcquisition:
                break

            # Do not even read if paused, unless we are caching the background, then
            # we still want to read.
            if self._pause and not self._caching_background:
                continue

            # image = np.rot90(self.screen.get_image_raw(), 3)
            image = self.screen.get_image_raw()
            if image is None:
                # this can happen sometimes, sometimes when switching cameras
                # get_image_raw can for a moment start returning None, of course
                # we need to account for this possibility and go next.
                continue
            
            # The convention is that we always put the current along the x-axis.
            image = np.rot90(image, 2)

            if self._caching_background:
                image_payload = self._handle_background(image)
            else:
                image_payload = self._do_image_analysis(image)

            self._push_to_subscribers(image_payload)

            if not self._pause:
                self.display_image_signal.emit(image_payload)

    def _push_to_subscribers(self, image_payload: ImagePayload) -> None:
        with self._lock:
            to_remove = []
            for imq in self._subscriber_queues:
                if not imq.num_images_remaining:
                    continue
                imq.q.put(image_payload)
                imq.num_images_remaining -= 1
                if imq.num_images_remaining == 0:
                    self._subscriber_queues.remove(imq)


    def _handle_background(self, image) -> ImagePayload:
        self.bg_images.appendleft(image)
        nbg_taken = len(self.bg_images)
        self.nbackground_taken.emit(nbg_taken)

        if nbg_taken == self.bg_images.maxlen:
            # Then we've filled the ring buffer of background images and
            # can go back to taking beam data, but first cache the mean_bg.
            self._caching_background = False
            self.mean_bg = np.mean(self.bg_images, axis=0, dtype=image.dtype)
            self.finished_background_acquisition.emit()

        return ImagePayload(
            image=image,
            screen_md=self.screen_md,
            time_calibration=self._time_calibration,
            bunch_charge=self._bunch_charge,
            is_bg=True
        )
    
    @property
    def read_frequency(self) -> float:
        return 1 / self._read_period
    
    def subscribe(self, nimages: int = 0) -> ImageQueue:
        imq = ImageQueue(queue.Queue(maxsize=nimages), num_images_remaining=nimages)
        self._subscriber_queues.append(imq)
        return imq

    def _take_n_fast(self, n: int) -> None:
        # This completely blocks the thread in that it is no longer listening
        # For messages.
        out = []
        assert n > 0
        for i in range(n):
            image = self.screen.get_image_raw()
            out.append(image)
            self.n_fast_state.emit(i)
            self._process_and_emit_image_for_display(image)
        # At least emit one image for display.
        self._process_and_emit_image_for_display(image)
        return out

    def submit(self, message: ImagingMessage) -> None:
        self._consumerq.put(message)

    def _dispatch_from_message_queue(self) -> None:
        # Check for messages from our parent
        # Only messages I account for are switching screen name
        # And switching image taking mode.  We consume all that have been queued
        # before proceeding with image taking.
        while True:
            try:
                message = self._consumerq.get(block=False)
            except queue.Empty:
                return

            if message is None:
                LOG.critical(
                    "%s received request to stop image acquisition", type(self).__name__
                )
                raise StopImageAcquisition("Image Acquisition halt requested.")

            self._handle_message(message)

    def _handle_message(self, message: ImagingMessage) -> None:
        result = None
        match message.mtype:
            # Could improve this pattern matching here by using both args of ImagingMessage.
            case MessageType.CACHE_BACKGROUND:
                self._caching_background = True
                self._clear_bg_cache()
                time.sleep(0.5)
            case MessageType.CLEAR_BACKGROUND:
                self._clear_bg_cache()
            # case MessageType.TAKE_N_FAST:
            #     result = self._take_n_fast(message.data["n"])
            case MessageType.CHANGE_SCREEN:
                self._set_screen(message.data["screen"])
            case MessageType.SUBTRACT_BACKGROUND:
                self._do_subtract_background = message.data["state"]
            case MessageType.PAUSE_SCREEN:
                self._pause = True
            case MessageType.PLAY_SCREEN:
                self._pause = False
            case MessageType.SET_FREQUENCY:
                self._read_period = 1 / message.data["frequency"]
            case MessageType.CLIP_OFFAXIS:
                self._clip_offaxis = message.data["state"]
            case MessageType.SMOOTH_IMAGE:
                self._smooth_image = message.data["state"]
            case MessageType.TIME_CALIBRATION:
                self._time_calibration = message.data["time_calibration"]
                self._bunch_charge = message.data["bunch_charge"]
            case _:
                message = f"Unknown message sent to {type(self).__name__}: {message}"
                LOG.critical(message)
                raise ValueError(message)
        self._consumerq.task_done()

    def _clear_bg_cache(self) -> None:
        self.bg_images = deque(maxlen=self.bg_images.maxlen)
        self.nbackground_taken.emit(0)

    def _do_image_analysis(self, image: npt.NDArray) -> None:
        if self.bg_images and self._do_subtract_background:
            image = image.astype(np.float64)
            image -= self.mean_bg
            image.clip(min=0, out=image)

        if self._smooth_image:
            Hg = ndimage.gaussian_filter(image, sigma=5)
            threshold = 0.05
            inds_mask_neg = np.asarray((Hg - np.max(Hg) * threshold) < 0).nonzero()
            image[inds_mask_neg] = 0.0

        
        if self._clip_offaxis:
            (xmin, xmax), (ymin, ymax) = self._clipping_bounds()
            zero_off_axis_regions(image, len(image) - xmax, len(image) - xmin, ymin, ymax)

        return ImagePayload(
            image=image,
            screen_md=self.screen_md,
            time_calibration=self._time_calibration,
            bunch_charge=self._bunch_charge,
            is_bg = False
        )

    @cache
    def _clipping_bounds(self) -> tuple[tuple[int, int], tuple[int, int]]:
        xminmax = self.screen.analysis.get_xroi_clipping()
        yminmax = self.screen.analysis.get_yroi_clipping()
        return xminmax, yminmax

    def _read_from_screen(self) -> npt.NDArray:
        image = self.screen.get_image_raw()
        if self._caching_background:
            self.bg_images.appendleft(image)
            if len(self.bg_images) == self.bg_images.maxlen:
                # Then we've filled the ring buffer of background images and
                # can go back to taking beam data, but first cache the mean_bg.
                self._caching_background = False
                self.mean_bg = np.mean(self.bg_images, axis=0, dtype=image.dtype)
        return image

    def _set_screen(self, screen: Screen) -> None:
        self.screen = screen
        self._clipping_bounds.cache_clear()
        self._clear_bg_cache()
        self._try_and_set_screen_metadata()
        self._xyflip = screen.get_hflip(), screen.get_vflip()

    def _try_and_set_screen_metadata(self):
        # if the screen is not powered, then getting the metadata will fail.
        # We will have to just try to get it in the future, hoping that at some
        # point some other part of the program will switch the screen on.
        # So we punt this into the long grass, so to speak.
        try:
            self.screen_md = self.screen.get_screen_metadata()
        except DOOCSReadError:
            timeout = 500
            LOG.warning(
                "Unable to acquire screen metadata, will try again in %ss",
                timeout / 1000,
            )
            QTimer.singleShot(timeout, self._try_and_set_screen_metadata)

    def change_screen(self, new_screen: Screen) -> None:
        message = ImagingMessage(MessageType.CHANGE_SCREEN, {"screen": new_screen})
        self._consumerq.put(message)
