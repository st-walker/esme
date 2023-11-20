from typing import Optional
import logging

from .dint import XFELMachineInterface
from .screens import ScreenService

LOG = logging.getLogger(__name__)

@datclass
class VoltageSample:
    address: str
    vmin: float
    vmax: float
    n: int


class DispersionMeasurer:
    PAUSE_BETWEEN_VOLTAGE_SETPOINTS: float = 1.0
    def __init__(self, ss: ScreenService,
                 di: Optional[XFELMachineInterface] = None) -> None:
        self.ss = ss
        self.di = di if di else XFELMachineInterface()

    def measure(self, screen_name):
        pixel_size = ss.get_pixel_dimensions()

        centres_of_mass = []
        # energies = []...
        for voltage in voltages:
            self.di.set_voltage(voltage)
            time.sleep(self.PAUSE_BETWEEN_VOLTAGE_SETPOINTS)
            com_bend_plane = self.get_centre_of_mass(screen_name)
            centres_of_mass.append(com_bend_plane)

        # do the fit, energies on x, centre of mass on y.
        dispersion = 3.0, 0.1 # with uncertainty...
        return dispersion

    def get_centres(self, screen_name):
        screen = self.self.

    def get_centre_of_mass(self, screen_name):
        image = self.ss.get_image(screen_name)
        image = process_image(image)
        com = centre_of_mass(image)

    def set_voltage(self, voltage):
        self.di.set_value("hello", voltage)
