import pytest
import unittest.mock as mock

from esme.control.kickers import FastKicker, FastKickerController, PolarityType, KickerOperationError, FastKickerSetpoint
from esme.control import EuXFELUserError

# if not is_in_control_room():
#     pytest.skip(allow_module_level=True, msg="Skipping because not in control room")



def test_kicker_is_hv_on(mocked_kicker):
    mocked_kicker.mi.get_value.return_value = FastKicker.HV_ON_READ_PROP_VALUE_ON
    is_on = mocked_kicker.is_hv_on()
    channel = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.HV_ON_READ_PROP_VALUE_ON}"
    mocked_kicker.mi.get_value.assert_called_once_with(channel)
    assert is_on == True

def test_kicker_is_hv_on(mocked_kicker):
    mocked_kicker.mi.get_value.return_value = FastKicker.HV_ON_READ_PROP_VALUE_ON
    is_on = mocked_kicker.is_hv_on()
    channel = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.HV_ON_READ_PROP}"
    mocked_kicker.mi.get_value.assert_called_once_with(channel)
    assert is_on == True

def test_kicker_set_hv_on(mocked_kicker):
    mocked_kicker.set_hv_on()
    channel = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.HV_EIN_SET_PROP}"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, 1)

def test_kicker_set_hv_off(mocked_kicker):
    mocked_kicker.set_hv_off()
    channel = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.HV_AUS_SET_PROP}"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, 1)

def test_kicker_raises_when_setting_polarity_whilst_hv_is_on(mocked_kicker):
    with mock.patch.object(FastKicker, "is_hv_on") as mocked_is_hv_on:
        mocked_is_hv_on.return_value = True
        with pytest.raises(KickerOperationError):
            mocked_kicker.set_polarity(PolarityType.POSITIVE)

def test_kicker_set_polarity_positive(mocked_kicker):
    mocked_kicker.set_polarity(PolarityType.POSITIVE)
    ch = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.POSITIVE_SET_PROP}"
    mocked_kicker.mi.set_value.assert_called_once_with(ch, 1)

def test_kicker_set_polarity_negative(mocked_kicker):
    ch = f"XFEL.SDIAG/ADIO24/SD54I1.0/{FastKicker.NEGATIVE_SET_PROP}"
    mocked_kicker.set_polarity(PolarityType.NEGATIVE)
    mocked_kicker.mi.set_value.assert_called_once_with(ch, 1)

def test_kicker_set_polarity_unknown_raises_typeerror(mocked_kicker):
    with pytest.raises(TypeError):
        mocked_kicker.set_polarity(object())
    
def test_kicker_set_delay(mocked_kicker):
    delay = 30
    mocked_kicker.set_delay(delay)
    channel = f"XFEL.SDIAG/TIMER/{mocked_kicker.trigger_channel}.DELAY"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, delay)

def test_kicker_set_delay_raises_on_negative_delay(mocked_kicker):
    delay = -30
    with pytest.raises(ValueError):
        mocked_kicker.set_delay(delay)

def test_kicker_set_delay_raises_on_negative_voltage(mocked_kicker):
    voltage = -30
    with pytest.raises(ValueError):
        mocked_kicker.set_voltage(voltage)
        
def test_kicker_set_voltage(mocked_kicker):
    voltage = 4000.
    mocked_kicker.set_voltage(voltage)
    channel = f"XFEL.SDIAG/KICKER.PS/{mocked_kicker.name}/S0"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, voltage)

def test_kicker_get_number(mocked_kicker):
    return_number = 3
    mocked_kicker.mi.get_value.return_value = return_number
    result = mocked_kicker.get_number()
    ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{mocked_kicker.name}/KICKER_NUMBER"
    mocked_kicker.mi.get_value.assert_called_once_with(ch)
    assert result == return_number

def test_kicker_is_operational(mocked_kicker):
    mocked_kicker.mi.get_value.return_value = 0 # 0 means good according to doocs explorer
    result = mocked_kicker.is_operational()
    ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{mocked_kicker.name}/KICKER_STATUS"
    mocked_kicker.mi.get_value.assert_called_once_with(ch)
    assert result == True

def test_FastKickerController_kicker_names(mocked_kicker_controller, mocked_kicker):
    assert mocked_kicker_controller.kicker_names == [mocked_kicker.name]

def test_FastKickerController_get_kicker_raise_on_unknown_kicker_name(mocked_kicker_controller):
    with pytest.raises(KickerOperationError):
        mocked_kicker_controller.get_kicker("a very made up kicker name")
