import numpy as np
import yaml
import pytest
import toml

from esme.control.mint import XFELMachineInterface
from esme.control.configs import build_simple_machine_from_config, load_kickers_from_config, load_screens_from_config, load_deflectors_from_config, load_calibration
from esme.control.kickers import PolarityType, FastKicker, FastKickerSetpoint

SIMPLE_LPS_CONFIG = """
kickers:
- name: KAX.54.I1
  adio24_fdl: XFEL.SDIAG/ADIO24/SD54I1.0
  trigger_channel: "SD54I1/RTM.TRG1"
  

screens:
- name: OTRC.55.I1
  area: I1
  kickers:
    KAX.54.I1:
      polarity: -1
      voltage: 9000
      delay: 46

- name: OTRB.450.B2
  area: B2
  kickers:
    KDY.445.B2:
      voltage: 11_500
      delay: 29
    KDY.446.B2:
      voltage: 11_500
      delay: 29

- name: OTRC.64.I1D
  area: I1

deflectors:
  - area: I1
    sp_fdl: XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/
    rb_fdl: XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/
  - area: B2
    sp_fdl: XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/
    rb_fdl: XFEL.RF/LLRF.CONTROLLER/VS.LLTDSB2/

"""

BOLKO_CALIBRATION = """
type = "bolko"
amplitudes = [8,  24]
slopes = [204, 670]
slope_units = "um/ps"
screen_name = "OTRC.64.I1D"
r34 = -10e-3
energy = 130e-3
frequency = 3e9
"""

IGOR_CALIBRATION = """
type = "igor"
amplitudes = [8, 18]
voltages = [0.38E6, 0.84E6]
"""

@pytest.fixture(scope="session")
def tmpconf(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "testconfig.yml"
    with open(filename, "w") as f:
        f.write(SIMPLE_LPS_CONFIG)
    return filename

@pytest.fixture(scope="session")
def tmpbcalib(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "tmpicalib.toml"
    with open(filename, "w") as f:
        f.write(BOLKO_CALIBRATION)
    return filename

@pytest.fixture(scope="session")
def tmpicalib(tmp_path_factory):
    filename = tmp_path_factory.mktemp("data") / "tmpicalib.toml"
    with open(filename, "w") as f:
        f.write(IGOR_CALIBRATION)
    return filename

@pytest.fixture
def loaded_conf(tmpconf) -> dict:
    with tmpconf.open("r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def loaded_bolko_calib(tmpbcalib):
    with tmpconf.open("r") as f:
        return toml.load(tmpbcalib)

@pytest.fixture
def tmp_igor_calib(tmpicalib):
    with tmpconf.open("r") as f:
        return toml.load(tmpicalib)
    

def test_load_kickers_from_config(loaded_conf):
    kicker_controller = load_kickers_from_config(loaded_conf)
    assert len(kicker_controller.kickers) == 1
    kicker = kicker_controller.kickers[0]
    assert kicker.name == "KAX.54.I1"
    assert kicker.adio24_fdl == "XFEL.SDIAG/ADIO24/SD54I1.0"
    assert kicker.trigger_channel == "SD54I1/RTM.TRG1"
    assert isinstance(kicker.mi, XFELMachineInterface)

def test_load_screens_from_config(loaded_conf):
    screen_service = load_screens_from_config(loaded_conf)

    assert screen_service.screen_names == ['OTRC.55.I1', 'OTRB.450.B2', 'OTRC.64.I1D']

    screen55 = screen_service.get_screen("OTRC.55.I1")
    screen55_ksps = screen55.fast_kicker_setpoints


    # Screen with one kicker setpoint
    assert screen55.name == "OTRC.55.I1"
    assert screen55_ksps == [FastKickerSetpoint(name="KAX.54.I1",
                                                polarity=PolarityType.NEGATIVE,
                                                voltage=9000,
                                                delay=46)]
    assert len(screen55_ksps) == 1

    screen450 = screen_service.get_screen("OTRB.450.B2")
    screen450_ksps = screen450.fast_kicker_setpoints
    assert screen450_ksps == [FastKickerSetpoint(name="KDY.445.B2",
                                                 polarity=None,
                                                 voltage=11_500,
                                                 delay=29),
                              FastKickerSetpoint(name="KDY.446.B2",
                                                 polarity=None,
                                                 voltage=11_500,
                                                 delay=29)]

    # Screen with no kickers (on axis only)
    screen64 = screen_service.get_screen("OTRC.64.I1D")
    assert screen64.name == "OTRC.64.I1D"
    assert screen64.fast_kicker_setpoints is None

def test_load_deflectors_from_config(loaded_conf):
    deflectors = load_deflectors_from_config(loaded_conf)

    ldeflectors = deflectors.deflectors

    assert len(ldeflectors) == 2

    assert ldeflectors[0].rb_fdl == "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/"
    assert ldeflectors[0].sp_fdl == "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/"
    assert ldeflectors[0].location == "I1"

    assert ldeflectors[1].rb_fdl == "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSB2/"
    assert ldeflectors[1].sp_fdl == "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/"
    assert ldeflectors[1].location == "B2"


def test_load_calibration_loads_bolko_style_calibration(tmpbcalib):
    calib = load_calibration(tmpbcalib)
    assert len(calib.setpoints) == 2


@pytest.mark.filterwarnings("ignore:Covariance of the parameters could not be estimated")
def test_load_calibration_loads_igor_style_calibration(tmpicalib):
    calib = load_calibration(tmpicalib)
    assert np.isclose(calib.get_voltage(8), 0.38E6)
    assert np.isclose(calib.get_amplitude(0.38E6), 8)    

    
