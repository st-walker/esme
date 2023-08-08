import copy
from unittest.mock import patch, MagicMock, call
from pathlib import Path

import pytest
import pandas as pd

from esme.measurement import (QuadrupoleSetting,
                              DispersionScanConfiguration,
                              TDSScanConfiguration,
                              ScreenPhotographer,
                              MeasurementRunner,
                              EnergySpreadMeasuringMachine,
                              MPS,
                              SNAPSHOT_TEMPL,
                              BasicDispersionMeasurer)

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
    beam_shots = 5
    bg_shots = 2
    result = photographer.take_data(bg_shots, beam_shots, delay=0)
    assert photographer.mps.beam_on.call_count == 2
    nshots = bg_shots + beam_shots
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
    reference_amplitude = 17
    scan_amplitudes = [11, 14, 17]
    scan_dispersion = 1.2
    return TDSScanConfiguration(reference_amplitude, scan_amplitudes, scan_dispersion)

@pytest.fixture
def mocked_runner(quad_scan_config, tscan_config, tmp_path):
    return MeasurementRunner("basename",
                             quad_scan_config,
                             tscan_config,
                             machine=MagicMock(EnergySpreadMeasuringMachine, name="Machine_mock"),
                             mps=MagicMock(MPS, name="MPS_mock"),
                             outdir=tmp_path)

def test_MeasurementRunner_init(quad_scan_config, tscan_config):
    name = "basename"
    runner = MeasurementRunner(name, quad_scan_config, tscan_config)

    assert runner.basename == name
    assert runner.quad_config == quad_scan_config
    assert runner.tds_config == tscan_config
    assert runner.outdir == Path("./")
    assert runner.machine.snapshot == SNAPSHOT_TEMPL
    # assert runner.photographer.mps == MPS()

# I want to mock set_reference_quads because I am testing it
# elsewhere.  here I want it to basically do nothing other than
# confirm that it's been called!

# @patch.object(MeasurementRunner, "make_df_filename", return_value="mocked-df-filename")
@patch.object(MeasurementRunner, "measure_reference_quads_dispersion", return_value=(1.2, 0.1))
def test_MeasurementRunner_tds_scan(mocked_mrqd, mocked_runner):
    # TDS scan args
    nbg = 4
    nbeam = 3
    delay = 0.01

    # Call the method I want to test
    outdir = mocked_runner.outdir
    # I mock make_df_filename because it relies
    with patch.object(MeasurementRunner, "make_df_filename", return_value=(outdir / "mocked_df_filename.pcl")):
        mocked_runner.tds_scan(nbg, nbeam, delay)
        from IPython import embed; embed()

    # measure_reference_quads_dispersion should be called once.  Always set the
    # correct quads for the TDS scan.
    mocked_mrqd.assert_called_once()

    # Assert set_tds_amplitude is called with the amplitudes stored.
    # Basically Confirms the TDS scan is indeed cycling through the
    # TDS amplitudes (voltages).
    amplitudes = mocked_runner.tds_config.scan_amplitudes
    mocked_runner.machine.set_tds_amplitude.call_args_list == [call(amp) for amp in amplitudes]

    from IPython import embed; embed()




def test_MeasurementRunner_set_reference_quads(mocked_runner):
    # Set the reference quads using the measurement runner.
    # from IPython import embed; embed()
    mocked_runner.set_reference_quads()
    # get reference QuadrupoleSetting instance
    ref_setting = mocked_runner.quad_config.reference_setting
    # Get names and strengths from the ref set.
    ref_quad_names = ref_setting.names
    ref_quad_strengths = ref_setting.strengths
    # Make call instances for each pair and assert that we correctly
    # called machine.set_quad with those pairs.
    calls = [call(name, strength) for name, strength in zip(ref_quad_names, ref_quad_strengths)]
    assert mocked_runner.machine.set_quad.call_args_list == calls

def test_TDSScanConfiguration(tscan_config):
    assert tscan_config.reference_amplitude == 17
    assert tscan_config.scan_amplitudes == [11, 14, 17]
    assert tscan_config.scan_dispersion == 1.2
