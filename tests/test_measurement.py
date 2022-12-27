import copy
from unittest.mock import patch, MagicMock, call

import pytest
import pandas as pd

from esme.measurement import (QuadrupoleSetting,
                              DispersionScanConfiguration,
                              TDSScanConfiguration,
                              ScreenPhotographer,
                              pop_df_row,
                              MeasurementRunner)


@pytest.fixture
def quad_setting():
    names = ["quad1", "quad2", "quad3"]
    strengths = [1, 2, 3]
    dispersion = 123
    return QuadrupoleSetting(names, strengths, dispersion)

@pytest.fixture
def quad_setting2():
    names = ["quad4", "quad5", "quad6"]
    strengths = [4, 5, 6]
    dispersion = 456
    return QuadrupoleSetting(names, strengths, dispersion)

@pytest.fixture
def quad_setting3():
    names = ["quad7", "quad8", "quad9"]
    strengths = [7, 8, 9]
    dispersion = 789
    return QuadrupoleSetting(names, strengths, dispersion)

def test_QuadrupoleSetting_init(quad_setting):
    assert quad_setting.names == ["quad1", "quad2", "quad3"]
    assert quad_setting.strengths == [1, 2, 3]
    assert quad_setting.dispersion == 123
    
def test_ScanOpticsConfiguration_init(quad_setting, quad_setting2, quad_setting3):
    inst = DispersionScanConfiguration(quad_setting, [quad_setting2, quad_setting3])

    assert inst.reference_setting == quad_setting
    assert inst.scan_settings == [quad_setting2, quad_setting3]

@pytest.fixture
def photographer():
    return ScreenPhotographer(mps=MagicMock(name="MPS_mock"),
                              machine=MagicMock(name="Machine_mock"))

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
    result = photographer.is_machine_offline()
    # Should call self.machine.get_machine_snapshot()
    machine_mock.is_machine_online.assert_called_once()
    assert result == False

@pytest.fixture
def quad_scan_config(quad_setting, quad_setting2, quad_setting3):
    return DispersionScanConfiguration(quad_setting, [quad_setting2, quad_setting3])

@pytest.fixture
def tscan_config():
    return TDSScanConfiguration(17, [11, 14, 17, 20], 1.2)

@pytest.fixture
def runner(quad_scan_config, tscan_config):
    return MeasurementRunner("basename", quad_scan_config, tscan_config, MagicMock(name="Machine_mock"))

def test_MeasurementRunner_set_reference_quads(runner):
    nref_quads = len(runner.quad_config.reference_setting.names)
    runner.set_reference_quads()
    assert runner.machine.set_quad.call_count == nref_quads
    assert runner.machine.set_quad.call_args_list == [call('quad1', 1),
                                                      call('quad2', 2),
                                                      call('quad3', 3)]

    
def test_TDSScanConfiguration(tscan_config):
    assert tscan_config.reference_amplitude == 17
    assert tscan_config.scan_amplitudes == [11, 14, 17, 20]
    assert tscan_config.scan_dispersion == 1.2 
