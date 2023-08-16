import pytest


from esme.control.exceptions import EuXFELUserError
from esme.control.screens import Screen, ScreenService
from esme.control.kickers import FastKicker, FastKickerController, PolarityType, KickerOperationError, FastKickerSetpoint


def test_ScreenService_get_image(mocked_screen_service):
    screen_name = "screen_name"
    return_value = object()
    mocked_screen_service.mi.get_value.return_value = return_value
    result = mocked_screen_service.get_image(screen_name)
    assert result is return_value
    ch = ScreenService.SCREEN_FDP_TEMPLATE.format(screen_name)
    mocked_screen_service.mi.get_value.assert_called_once_with(ch)


def test_ScreenService_screen_names(mocked_screen_service, screen):
    assert mocked_screen_service.screen_names == [screen.name]

def test_ScreenService_get_screen(mocked_screen_service, screen):
    got_screen = mocked_screen_service.get_screen(screen.name)
    assert got_screen == screen

def test_ScreenService_raises_when_unrecognised_screen(mocked_screen_service, screen):
    with pytest.raises(ValueError):
        got_screen = mocked_screen_service.get_screen("UNKNOWABLE_SCREEN_NAME")

def test_ScreenService_get_fast_kicker_setpoints_for_screen(mocked_screen_service, fast_kicker_sp, screen):
    sps = mocked_screen_service.get_fast_kicker_setpoints_for_screen(screen.name)
    assert [fast_kicker_sp] == sps

def test_ScreenService_get_fast_kicker_sp_for_screen_raises_when_screen_has_no_kicker_sp_info(mocked_screen_service, fast_kicker_sp, screen):
    sp = mocked_screen_service.get_fast_kicker_setpoints_for_screen(screen.name)
    
# def test_Screen_is_in(screen):
    
    
