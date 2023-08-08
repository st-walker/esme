from unittest.mock import patch, MagicMock, call
from esme.tds import TDSController, TDSAddresses, SetpointReadbackPair

import pytest


@pytest.fixture
def i1_addresses():
    return TDSAddresses(
        amplitude=SetpointReadbackPair(setpoint="asp", readback="arb"),
        phase=SetpointReadbackPair(setpoint="psp", readback="prb"),
        event="event_channel",
        bunch_one="b1_channel",
    )


@pytest.fixture
def mtds(i1_addresses):
    machine_mock = MagicMock(name="Machine_mock")
    return TDSController(i1_addresses, mi=machine_mock)


def test_TDSController_set_amplitude(mtds):
    value = 3
    address = mtds.addies.amplitude.setpoint
    mtds.set_amplitude(value)
    mtds._mi.set_value.assert_called_with(address, value)


def test_TDSController_read_rb_amplitude(mtds):
    value = 3
    address = mtds.addies.amplitude.readback
    mtds._mi.get_value.return_value = value
    assert mtds.read_rb_amplitude() == value
    mtds._mi.get_value.assert_called_with(address)


def test_TDSController_read_sp_amplitude(mtds):
    value = 3
    address = mtds.addies.amplitude.setpoint
    mtds._mi.get_value.return_value = value
    assert mtds.read_sp_amplitude() == value
    mtds._mi.get_value.assert_called_with(address)


def test_TDSController_set_phase(mtds):
    phase = 4
    address = mtds.addies.phase.setpoint
    mtds.set_phase(phase)
    mtds._mi.set_value.assert_called_with(address, phase)


def test_TDSController_read_rb_phase(mtds):
    expected_phase = 3
    address = mtds.addies.phase.readback
    mtds._mi.get_value.return_value = expected_phase
    assert mtds.read_rb_phase() == expected_phase
    mtds._mi.get_value.assert_called_with(address)


def test_TDSController_read_on_beam_timing(mtds):
    expected_bunch_one = 3
    address = mtds.addies.bunch_one
    mtds._mi.get_value.return_value = expected_bunch_one
    assert mtds.read_on_beam_timing() == expected_bunch_one
    mtds._mi.get_value.assert_called_with(address)


# def test_TDSController_read_sp_amplitude(mtds):
#     value = 3
#     address = mtds.addies.amplitude_setpoint
#     mtds._mi.get_value.return_Value

# def test_TDSController_read_sp_amplitude(mtds):
#     value = 3
#     address = mtds.addies.amplitude_setpoint
#     mtds._mi.get_value.return_value = value
#     assert mtds.read_rb_amplitude() == value

# mtds._mi.set_value.assert_called_with(address, value)
