import pytest
import unittest.mock as mock

from esme.control.kickers import ScreenConfig, Kicker, KickerController, PolarityType, KickerOperationError
from esme.control import EuXFELUserError

# if not is_in_control_room():
#     pytest.skip(allow_module_level=True, msg="Skipping because not in control room")


ADIO24_STEM = "XFEL.SDIAG/ADIO24/SD54I1.0"
TRIGGER_CHANNEL = "SD54I1/RTM.TRG1"


@pytest.fixture
def screen_config():
    kwargs = {'name': 'OTRC.55.I1',
              'polarity': PolarityType.NEGATIVE,
              'voltage': 9.0,
              'delay': 46,
              'kicker_name': 'KAX.54.I1'}

    return ScreenConfig(**kwargs)


@pytest.fixture
def mocked_kicker(mock_machine):
    return Kicker(name="KAX.54.I1",
                  adio24_stem=ADIO24_STEM,
                  trigger_channel=TRIGGER_CHANNEL,
                  mi=mock_machine)

@pytest.fixture
def mocked_kicker_controller(screen_config, mocked_kicker, mock_machine):
    return KickerController([screen_config], [mocked_kicker], mi=mock_machine)

def test_kicker_is_hv_on(mocked_kicker):
    mocked_kicker.mi.get_value.return_value = Kicker.HV_ON_READ_VALUE_ON
    is_on = mocked_kicker.is_hv_on()
    channel = f"{ADIO24_STEM}/{Kicker.HV_ON_READ}"
    mocked_kicker.mi.get_value.assert_called_once_with(channel)
    assert is_on == True

def test_kicker_is_hv_on(mocked_kicker):
    mocked_kicker.mi.get_value.return_value = Kicker.HV_ON_READ_VALUE_ON
    is_on = mocked_kicker.is_hv_on()
    channel = f"{ADIO24_STEM}/{Kicker.HV_ON_READ}"
    mocked_kicker.mi.get_value.assert_called_once_with(channel)
    assert is_on == True

def test_kicker_set_hv_on(mocked_kicker):
    mocked_kicker.set_hv_on()
    channel = f"{ADIO24_STEM}/{Kicker.HV_EIN_SET}"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, 1)

def test_kicker_set_hv_off(mocked_kicker):
    mocked_kicker.set_hv_off()
    channel = f"{ADIO24_STEM}/{Kicker.HV_AUS_SET}"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, 1)

def test_kicker_raises_when_setting_polarity_whilst_hv_is_on(mocked_kicker):
    with mock.patch.object(Kicker, "is_hv_on") as mocked_is_hv_on:
        mocked_is_hv_on.return_value = True
        with pytest.raises(KickerOperationError):
            mocked_kicker.set_polarity(PolarityType.POSITIVE)

def test_kicker_set_polarity_positive(mocked_kicker):
    mocked_kicker.set_polarity(PolarityType.POSITIVE)
    mocked_kicker.mi.set_value.assert_called_once_with(Kicker.POSITIVE_SET, 1)

def test_kicker_set_polarity_negative(mocked_kicker):
    mocked_kicker.set_polarity(PolarityType.NEGATIVE)
    mocked_kicker.mi.set_value.assert_called_once_with(Kicker.NEGATIVE_SET, 1)

def test_kicker_set_delay(mocked_kicker):
    delay = 30
    mocked_kicker.set_delay(delay)
    channel = f"XFEL.SDIAG/TIMER/{mocked_kicker.trigger_channel}.DELAY"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, delay)

def test_kicker_set_voltage(mocked_kicker):
    voltage = 4000.
    mocked_kicker.set_voltage(voltage)
    channel = f"XFEL.SDIAG/KICKER.PS/{mocked_kicker.name}/S0"
    mocked_kicker.mi.set_value.assert_called_once_with(channel, voltage)

def test_KickerController_screen_names(mocked_kicker_controller, screen_config):
    assert mocked_kicker_controller.screen_names == [screen_config.name]

def test_KickerController_kicker_names(mocked_kicker_controller, mocked_kicker):
    assert mocked_kicker_controller.kicker_names == [mocked_kicker.name]

def test_KickerController_configure_kicker_for_screen_raises_on_bad_screen_name(mocked_kicker_controller):
    with pytest.raises(KickerOperationError):
        mocked_kicker_controller.configure_kicker_for_screen("badname")

def test_KickerController_configure_kicker_for_screen_configures_screen(mocked_kicker_controller, screen_config):
    with (mock.patch.object(Kicker, "set_hv_off"),
          mock.patch.object(Kicker, "set_hv_on"),
          mock.patch.object(Kicker, "set_polarity"),
          mock.patch.object(Kicker, "set_voltage"),
          mock.patch.object(Kicker, "set_delay")):
        mocked_kicker_controller.configure_kicker_for_screen(screen_config.name)

        Kicker.set_hv_off.assert_called_once_with()
        Kicker.set_polarity.assert_called_once_with(screen_config.polarity)
        Kicker.set_delay.assert_called_once_with(screen_config.delay)
        Kicker.set_voltage.assert_called_once_with(screen_config.voltage)
        Kicker.set_hv_on.assert_called_once_with()        
