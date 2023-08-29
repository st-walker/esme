import numpy as np
import pytest

from esme.gui.scannerpanel import ScanWorker
from esme.gui.common import build_default_lps_machine
from esme.control.configs import load_calibration
from esme.calibration import DiscreteCalibration


fname = "igor-conf.toml"

@pytest.mark.slow_integration_test
def test_energy_spread_2021_calculation():
    machine = build_default_lps_machine()
    calibration = load_calibration(fname)
    voltages = [0.38E6, 0.47E6, 0.56E6, 0.61e6, 0.65E6, 0.75E6]
    calibration = DiscreteCalibration(calibration.amplitudes, voltages)
    worker = ScanWorker(machine, calibration,
                        voltages=calibration.voltages,
                        slug="test-simulation")
    result = worker.run()
    assert np.isclose(result.sigma_e[0], 5738.279453892163)


