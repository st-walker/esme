import pytest

from esme.control.dint import DOOCS_FULL_ADDRESS_REGEX


# fmt: off
@pytest.mark.parametrize("test_string,matches", [
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", True),
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL/", False),
    ("XFEL.DIAG/BEAM*ENERGY_MEASUREMENT/I1D/ENERGY.ALL", True),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL/", True),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL*", True),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL*/", True),
    ("XFEL*.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", True),
    # 5 fields, but we only allow 4
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL/EXTRA", False),
    # Invalid, lower case letters
    ("invalid.characters/here/I1D/ENERGY.ALL", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY*.ALL", True),
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL*", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/*ENERGY.ALL", True),
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/*ENERGY.ALL/", False),
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL*/", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY/ALL", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL/EXTRA*/", False),
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY*/ALL/", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/EN*ERGY.ALL/", True),
    # Invalid: non-alphanumeric character '@'
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY_ALL@", False),
    # Invalid: non-alphanumeric character '$'
    ("XFEL.DIAG/BEAM$ENERGY_MEASUREMENT/I1D/ENERGY.ALL", False),
    # Invalid: non-alphanumeric character '!'
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY!ALL", False),
    # Invalid: non-alphanumeric character '#'
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL#", False),
    # Invalid with empty section
    ("/XFEL.DIAG/BEAM_ENERGY_MEASUREMENT//I1D/ENERGY.ALL", False),
    # Invalid with leading empty section
    ("//XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", False),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL//", False),
    ("///", True),
    ("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT///", True),
])
def test_full_doocs_address_pattern(test_string, matches):
    assert bool(DOOCS_FULL_ADDRESS_REGEX.match(test_string)) is matches
# fmt: on
