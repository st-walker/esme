from unittest import mock

import pytest

from esme.control.scanner import Scanner, ScanConfig, QuadScanSetpoint, ScanSetpointError
from esme.control.sbunches import DiagnosticRegion
from esme.control.vmint import DictionaryXFELMachineInterface

@pytest.fixture
def scan_setpoint():
    k1ls = {
        'QI.52.I1': -0.0837203,
        'QI.53.I1': 0.500308,
        'QI.54.I1': 0.18882,
        'QI.55.I1': -0.71248,
        'QI.57.I1': 0.71248,
        'QI.59.I1': -0.71248,
        'QI.60.I1': -0.509559,
        'QI.61.I1': 0.83263,
        'QI.63.I1D': -0.249585,
        'QI.64.I1D': 0.83195,
    }
    dispersion = 1.2
    return QuadScanSetpoint(k1ls, dispersion)


@pytest.fixture
def scan_config(scan_setpoint):
    return ScanConfig("scan name", [scan_setpoint], 0.5, tds_scan_dispersion=1.2, area="I1")


@pytest.fixture
def mocked_scanner(scan_config, mock_machine):
    return Scanner(scan_config, mi=mock_machine)


def test_mocked_scanner_set_quad_strength(mocked_scanner):
    quad_name = "QI.52.I1"
    kmrad = 0.423
    mocked_scanner.set_quad_strength(quad_name, kmrad)
    ch = f"XFEL.MAGNETS/MAGNET.ML/{quad_name}/KICK_MRAD.SP"
    mocked_scanner.mi.set_value.assert_called_once_with(ch, kmrad)


def test_mocked_scanner_get_quad_strength(mocked_scanner):
    quad_name = "QI.52.I1"
    kmrad = 0.423
    mocked_scanner.mi.get_value.return_value = kmrad
    ch = f"XFEL.MAGNETS/MAGNET.ML/{quad_name}/KICK_MRAD.SP"
    actual_kmrad = mocked_scanner.get_quad_strength(quad_name)
    mocked_scanner.mi.get_value.assert_called_once_with(ch)
    assert actual_kmrad == kmrad


def test_mocked_scanner_get_setpoint_raises_on_bad_dispersion(mocked_scanner):
    with pytest.raises(ValueError):
        dispersion = 1e6
        mocked_scanner.get_setpoint(dispersion)

def test_mocked_scanner_get_setpoint(mocked_scanner, scan_setpoint):
    dispersion = 1.2
    returned_setpoint = mocked_scanner.get_setpoint(dispersion)
    assert returned_setpoint == scan_setpoint

def test_mocked_scanner_set_scan_setpoint_quads(mocked_scanner):
    # TODO: flesh this out with some assertions...
    mocked_scanner.set_scan_setpoint_quads(1.2)

def test_mocked_scanner_infer_intended_dispersion_setpoint(mocked_scanner, scan_setpoint):
    # So we have some stored state for simple checking of setting,
    # easier than using Mock in this instance...
    mocked_scanner.mi = DictionaryXFELMachineInterface()
    mocked_scanner.set_scan_setpoint_quads(1.2)
    assert scan_setpoint == mocked_scanner.infer_intended_dispersion_setpoint()

def test_mocked_scanner_infer_intended_dispersion_setpoint_raises(mocked_scanner):
    mocked_scanner.mi = DictionaryXFELMachineInterface()
    # The machine now has some state with this:
    mocked_scanner.set_scan_setpoint_quads(1.2)
    # Now we want to change the machine state slightly so when we try
    # to match the stored setpoitns against the machine we will get an
    # exception.
    quad_name = "QI.52.I1"
    kmrad = 0.423
    mocked_scanner.set_quad_strength(quad_name, kmrad)
    ch = f"XFEL.MAGNETS/MAGNET.ML/{quad_name}/KICK_MRAD.SP"
    mocked_scanner.mi.set_value(ch, kmrad)

    with pytest.raises(ScanSetpointError):
        mocked_scanner.infer_intended_dispersion_setpoint()
