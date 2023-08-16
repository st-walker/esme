import pytest
from unittest.mock import MagicMock

from esme.control import (FastKickerController, FastKicker,
                          Screen,
                          ScreenService, FastKickerSetpoint,
                          BunchLengthMachine, PolarityType,
                          TransverseDeflector,
                          TransverseDeflectors,
                          SpecialBunchesControl)


KICKER_ADIO24_FDL = "XFEL.SDIAG/ADIO24/SD54I1.0"
KICKER_TRIGGER_CHANNEL = "SD54I1/RTM.TRG1"
SCREEN_NAME = "OTRC.55.I1"

@pytest.fixture
def mock_machine():
    return MagicMock(name="Machine_mock")

@pytest.fixture
def fast_kicker_sp():
    kwargs = {'name': 'KAX.54.I1',
              'polarity': PolarityType.NEGATIVE,
              'voltage': 9000.,
              'delay': 46}

    return FastKickerSetpoint(**kwargs)


@pytest.fixture
def screen(fast_kicker_sp):
    return Screen(name=SCREEN_NAME,
                  location=SpecialBunchesControl("I1"),
                  fast_kicker_setpoints=[fast_kicker_sp])

@pytest.fixture
def mocked_screen_service(screen, mock_machine):
    return ScreenService([screen], mi=mock_machine)

@pytest.fixture
def mocked_kicker(mock_machine):
    return FastKicker(name="KAX.54.I1",
                      adio24_fdl=KICKER_ADIO24_FDL,
                      trigger_channel=KICKER_TRIGGER_CHANNEL,
                      mi=mock_machine)

@pytest.fixture
def mocked_kicker_controller(mocked_kicker, mock_machine):
    return FastKickerController([mocked_kicker], mi=mock_machine)

@pytest.fixture
def mocked_deflector(mock_machine):
    return TransverseDeflector(location="I1",
                               sp_fdl="XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/",
                               rb_fdl="XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/",
                               mi=mock_machine)

@pytest.fixture
def mocked_deflectors(mocked_deflector):
    return TransverseDeflectors(mocked_deflector)

@pytest.fixture
def mocked_bunch_length_machine(mocked_kicker_controller, mocked_screen_service, mock_machine, mocked_deflectors):
    return BunchLengthMachine(mocked_kicker_controller, mocked_screen_service, mocked_deflectors, mi=mock_machine)

@pytest.fixture
def mocked_sbc(mock_machine):
    return SpecialBunchesControl(mi=mock_machine)
