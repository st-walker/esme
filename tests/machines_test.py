import pytest
from unittest.mock import MagicMock
from unittest import mock

from esme.control.machines import BunchLengthMachine
from esme.control import FastKicker




def test_BunchLengthMachine_set_kicker_for_screen(mocked_bunch_length_machine, fast_kicker_sp):
    mocked_bunch_length_machine.set_kicker_for_screen("OTRC.55.I1")

def test_BunchLengthMachine_configure_kicker_for_screen(mocked_bunch_length_machine, screen, fast_kicker_sp):
    with (mock.patch.object(FastKicker, "set_hv_off"),
          mock.patch.object(FastKicker, "set_hv_on"),
          mock.patch.object(FastKicker, "set_polarity"),
          mock.patch.object(FastKicker, "set_voltage"),
          mock.patch.object(FastKicker, "set_delay")):
        mocked_bunch_length_machine.set_kicker_for_screen(screen.name)

        FastKicker.set_hv_off.assert_called_once_with()
        FastKicker.set_polarity.assert_called_once_with(fast_kicker_sp.polarity)
        FastKicker.set_delay.assert_called_once_with(fast_kicker_sp.delay)
        FastKicker.set_voltage.assert_called_once_with(fast_kicker_sp.voltage)
        FastKicker.set_hv_on.assert_called_once_with()
