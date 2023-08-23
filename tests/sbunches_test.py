import copy
from unittest import mock

import pytest

from esme.control.sbunches import DiagnosticRegion, SpecialBunchesControl
from esme.control.mint import XFELMachineInterface

# I got this from the control room, one manual read using pydoocs.read.
# In principle this can change but I guess not very often...
KICKER_NAME_TO_KICKER_INDEX_MAP = [[1, 1.0, 1.0, 1692034240, 'KAX.54.I1'],
                                   [2, 2.0, 2.0, 1692034240, 'KAX.56.I1'],
                                   [1, 1.0, 1.0, 1692034240, 'KAY.214.B1'],
                                   [2, 2.0, 2.0, 1692034240, 'KAY.216.B1'],
                                   [3, 3.0, 3.0, 1692034240, 'KAY.218.B1'],
                                   [4, 4.0, 4.0, 1692034240, 'KAY.219.B1'],
                                   [1, 1.0, 1.0, 1692034241, 'KDY.445.B2'],
                                   [1, 1.0, 1.0, 1692034241, 'KDY.446.B2'],
                                   [2, 2.0, 2.0, 1692034241, 'KDY.452.B2'],
                                   [2, 2.0, 2.0, 1692034241, 'KDY.453.B2']]


def test_SpecialBunchesControl_get_kicker_name_to_kicker_index_map(mocked_sbc):
    mocked_sbc.mi.get_value.return_value = KICKER_NAME_TO_KICKER_INDEX_MAP
    kmap = mocked_sbc.get_kicker_name_to_kicker_index_map()
    ch = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/K*/KICKER_NUMBER"
    mocked_sbc.mi.get_value.assert_called_once_with(ch)
    assert kmap == {'KAX.54.I1': 1,
                    'KAX.56.I1': 2,
                    'KAY.214.B1': 1,
                    'KAY.216.B1': 2,
                    'KAY.218.B1': 3,
                    'KAY.219.B1': 4,
                    'KDY.445.B2': 1,
                    'KDY.446.B2': 1,
                    'KDY.452.B2': 2,
                    'KDY.453.B2': 2}

def test_SpecialBunchesControl_get_kicker_control_list(mocked_sbc):
    fixed_result = [1, 2, 3, 4]
    mocked_sbc.mi.get_value.return_value = fixed_result
    result = mocked_sbc.get_kicker_control_list()
    assert result == fixed_result
    mocked_sbc.mi.get_value.assert_called_once_with(mocked_sbc.control_address())

@pytest.mark.parametrize("location,expected", [("I1", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL"),
                                           ("B2", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/B2/CONTROL")])
def test_SpecialBunchesControl_control_address(mocked_sbc, location, expected):
    mocked_sbc.location = location
    assert mocked_sbc.control_address() == expected

@pytest.mark.parametrize("location,expected", [("I1", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/SUBTRAIN"),
                                           ("B2", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/B2/SUBTRAIN")])
def test_SpecialBunchesControl_beamregion_address(mocked_sbc, location, expected):
    mocked_sbc.location = location
    assert mocked_sbc.beamregion_address() == expected

@pytest.mark.parametrize("location,expected", [("I1", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/PULSES.ACTIVE"),
                                           ("B2", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/B2/PULSES.ACTIVE")])
def test_SpecialBunchesControl_beamregion_address(mocked_sbc, location, expected):
    mocked_sbc.location = location
    assert mocked_sbc.npulses_address() == expected

def test_SpecialBunchesControl_set_beam_region(mocked_sbc):
    br = 4
    ch = mocked_sbc.beamregion_address()
    mocked_sbc.set_beam_region(br)
    mocked_sbc.mi.set_value.assert_called_once_with(ch, br)

def test_SpecialBunchesControl_set_npulses(mocked_sbc):
    npulses = 4
    ch = mocked_sbc.npulses_address()
    mocked_sbc.set_npulses(npulses)
    mocked_sbc.mi.set_value.assert_called_once_with(ch, npulses)

def test_SpecialBunchesControl_set_bunch_number(mocked_sbc):
    with mock.patch.object(SpecialBunchesControl, "get_kicker_control_list"):
        mocked_sbc.get_kicker_control_list.return_value = [0, 1, 2, 3]
        intended_bunch_number = 4
        mocked_sbc.set_bunch_number(intended_bunch_number)
        intended_control_arg = [intended_bunch_number, 1, 2, 3]
        mocked_sbc.mi.set_value.assert_called_once_with(mocked_sbc.control_address(),
                                                        intended_control_arg)

def test_SpecialBunchesControl_set_use_tds(mocked_sbc):
    with mock.patch.object(SpecialBunchesControl, "get_kicker_control_list"):
        mocked_sbc.get_kicker_control_list.return_value = [1, int(True), 2, 3]
        intended_use_tds = False
        mocked_sbc.set_use_tds(intended_use_tds)
        intended_control_arg = [1, int(intended_use_tds), 2, 3]
        mocked_sbc.mi.set_value.assert_called_once_with(mocked_sbc.control_address(),
                                                        intended_control_arg)

def test_SpecialBunchesControl_set_kicker(mocked_sbc):
    with (mock.patch.object(SpecialBunchesControl, "get_kicker_control_list"),
          mock.patch.object(SpecialBunchesControl, "get_kicker_name_to_kicker_index_map")):
        mocked_sbc.get_kicker_control_list.return_value = [1, 1, 10, 3]
        mocked_sbc.get_kicker_name_to_kicker_index_map.return_value = {"KAX.54.I1": 1,
                                                                       "KAX.56.I1": 2}
        kicker_name = "KAX.56.I1"
        intended_kicker_number = 2
        mocked_sbc.set_kicker_name(kicker_name)
        intended_control_arg = [1, 1, intended_kicker_number, 3]
        mocked_sbc.mi.set_value.assert_called_once_with(mocked_sbc.control_address(),
                                                        intended_control_arg)

def test_SpecialBunchesControl_status_address(mocked_sbc):
    location = DiagnosticRegion.I1
    mocked_sbc.location = location
    thing_to_be_checked = "TDS"
    expected = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/STATUS.{}".format(location, thing_to_be_checked)
    result = mocked_sbc.status_address(thing_to_be_checked)
    assert expected == result
    
@pytest.mark.parametrize("tds_status_read,expected", [(0, True), (1, False)])
def test_is_tds_ok(mocked_sbc, tds_status_read, expected):
    ch = mocked_sbc.status_address("TDS")
    mocked_sbc.mi.get_value.return_value = tds_status_read
    result = mocked_sbc.is_tds_ok()
    mocked_sbc.mi.get_value.assert_called_once_with(ch)
    assert result == expected

@pytest.mark.parametrize("screen_status_read,expected", [(0, True), (1, False)])
def test_is_screen_ok(mocked_sbc, screen_status_read, expected):
    ch = mocked_sbc.status_address("SCREEN")
    mocked_sbc.mi.get_value.return_value = screen_status_read
    result = mocked_sbc.is_screen_ok()
    mocked_sbc.mi.get_value.assert_called_once_with(ch)
    assert result == expected

@pytest.mark.parametrize("kicker_status_read,expected", [(0, True), (1, False)])
def test_is_kicker_ok(mocked_sbc, kicker_status_read, expected):
    ch = mocked_sbc.status_address("KICKER")
    mocked_sbc.mi.get_value.return_value = kicker_status_read
    result = mocked_sbc.is_kicker_ok()
    mocked_sbc.mi.get_value.assert_called_once_with(ch)
    assert result == expected
    

@pytest.mark.parametrize("location,expected", [("I1", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/START"),
                                           ("B2", "XFEL.SDIAG/SPECIAL_BUNCHES.ML/B2/START")])
def test_fire_diagnostic_bunch_address(mocked_sbc, location, expected):
    mocked_sbc.location = location
    result = mocked_sbc.fire_diagnostic_bunch_address()
    assert result == expected

    
