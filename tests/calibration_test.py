import pytest
import numpy as np

from esme.calibration import BolkoCalibrationSetPoint, TDSCalibration


@pytest.fixture
def bolko_calib():
    return BolkoCalibrationSetPoint(amplitude=8,
                                    slope=204e6,
                                    r34=-5.4,
                                    energy=130,
                                    frequency=3e9)


def test_bolko_calib_get_voltage(bolko_calib):
    voltage = bolko_calib.get_voltage() * 1e-6
    assert np.isclose(voltage, 0.2605425364689546)
