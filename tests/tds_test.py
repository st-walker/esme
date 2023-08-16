import pytest

from esme.control import TransverseDeflector


def test_deflector_get_phase_rb(mocked_deflector):
    rb = mocked_deflector.get_phase_rb()
    ch = f"{mocked_deflector.rb_fdl}{TransverseDeflector.PHASE_RB_PROP}"
    mocked_deflector.mi.get_value.assert_called_once_with(ch)

def test_deflector_get_amplitude_rb(mocked_deflector):
    rb = mocked_deflector.get_amplitude_rb()
    ch = f"{mocked_deflector.rb_fdl}{TransverseDeflector.AMPLITUDE_RB_PROP}"
    mocked_deflector.mi.get_value.assert_called_once_with(ch)

def test_deflector_get_phase_sp(mocked_deflector):
    rb = mocked_deflector.get_phase_sp()
    ch = f"{mocked_deflector.sp_fdl}{TransverseDeflector.PHASE_SP_PROP}"
    mocked_deflector.mi.get_value.assert_called_once_with(ch)

def test_deflector_get_amplitude_sp(mocked_deflector):
    rb = mocked_deflector.get_amplitude_sp()
    ch = f"{mocked_deflector.sp_fdl}{TransverseDeflector.AMPLITUDE_SP_PROP}"
    mocked_deflector.mi.get_value.assert_called_once_with(ch)

def test_deflector_set_amplitude(mocked_deflector):
    value = 5.0
    rb = mocked_deflector.set_amplitude(value)
    ch = f"{mocked_deflector.sp_fdl}{TransverseDeflector.AMPLITUDE_SP_PROP}"
    mocked_deflector.mi.set_value.assert_called_once_with(ch, value)

def test_deflector_set_phase(mocked_deflector):
    value = 5.0
    rb = mocked_deflector.set_phase(value)
    ch = f"{mocked_deflector.sp_fdl}{TransverseDeflector.PHASE_SP_PROP}"
    mocked_deflector.mi.set_value.assert_called_once_with(ch, value)
    
