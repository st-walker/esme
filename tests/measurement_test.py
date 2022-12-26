import copy
from unittest.mock import patch, MagicMock, call

import pytest
import pandas as pd

from esme.measurement import (QuadrupoleSetting,
                              InjectorScanOpticsConfiguration,
                              ScreenPhotographer,
                              pop_df_row)


@pytest.fixture
def quad_setting():
    names = ["quad1", "quad2", "quad3"]
    strengths = [1, 2, 3]
    dispersion = 0.5
    return QuadrupoleSetting(names, strengths, dispersion)

def test_QuadrupoleSetting_init(quad_setting):
    assert quad_setting.names == ["quad1", "quad2", "quad3"]
    assert quad_setting.strengths == [1, 2, 3]
    assert quad_setting.dispersion == 0.5

def test_ScanOpticsConfiguration_init(quad_setting):
    qs2 = copy.deepcopy(quad_setting)
    qs3 = copy.deepcopy(quad_setting)
    qs2.dispersion = 0.4
    qs3.dispersion = 0.2

    inst = InjectorScanOpticsConfiguration(quad_setting, [qs2, qs3])

    assert inst.reference_setting == quad_setting
    assert inst.scan_settings == [qs2, qs3]

@pytest.fixture
def photographer():
    return ScreenPhotographer(mps=MagicMock(name="MPS_mock"),
                              machine=MagicMock(name="Machine_mock"))

# @pytest.fixture
# def df():
#     return MagicMock(name="df_mock")

def test_ScreenPhotographer_switch_beam_on(photographer):
    photographer.switch_beam_on()
    photographer.mps.beam_on.assert_called_once()

def test_ScreenPhotographer_switch_beam_off(photographer):
    photographer.switch_beam_off()
    photographer.mps.beam_off.assert_called_once()

def test_ScreenPhotographer_take_data(photographer):
    nshots = 5
    result = photographer.take_data(nshots, delay=0)
    photographer.mps.beam_on.assert_called_once()
    assert len(result) == nshots
    assert photographer.machine.get_machine_snapshot.call_count == nshots

def test_ScreenPhotographer_take_background(photographer):
    nshots = 5
    mps_mock = photographer.mps
    
    result = photographer.take_background(nshots, delay=0)
    photographer.mps.beam_off.assert_called_once()
    photographer.mps.beam_on.assert_called_once()
    assert photographer.machine.get_machine_snapshot.call_count == nshots
    assert len(result) == nshots

def test_ScreenPhotographer_take_machine_snapshot(photographer):
    machine_mock = photographer.machine
    photographer.take_machine_snapshot()
    # Should call self.machine.get_machine_snapshot()
    machine_mock.get_machine_snapshot.assert_called_once()

def test_pop_df_row():
    df = pd.DataFrame({"A": [1,2,3], "B": [3,4,5]})
    df_new, row = pop_df_row(df, -1)
    df_new.loc[len(df_new)] = row
    assert (df_new == df).all().all()

def test_ScreenPhotographer_is_machine_off(photographer):
    machine_mock = photographer.machine
    machine_mock.is_machine_online.return_value = True
    result = photographer.is_machine_off()
    # Should call self.machine.get_machine_snapshot()
    machine_mock.is_machine_online.assert_called_once()
    assert result == False
