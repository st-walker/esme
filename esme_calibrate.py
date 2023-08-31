from esme.gui.calibrator import CalibrationWorker
import numpy as np

from esme.gui.scannerpanel import ScanWorker
from esme.gui.common import build_default_lps_machine
from esme.control.configs import load_calibration
from esme.calibration import DiscreteCalibration


machine = build_default_lps_machine()

calibrator = CalibrationWorker(machine, "OTRC.55.I1", amplitudes=[15])

calibrator.run()
