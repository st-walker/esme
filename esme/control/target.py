import logging



LOG = logging.getLogger(__name__)

FREQUENCY_COMMAND = "set_frequency_ratio 4;set_count 1 1"

TO_B2D_COMMAND = "reset_counts;configure_bunch_counter 1 B 0;configure_bunch_counter 2 . .;configure_bunch_counter 3 . .;configure_bunch_counter 4 . .;set_tail_cleanup;set_pattern_sequence [A];clear_patterns;add_subpattern B 500 BC2 dump;"
TO_I1D_COMMAND = "reset_counts;configure_bunch_counter 1 I 0;configure_bunch_counter 2 . .;configure_bunch_counter 3 . .;configure_bunch_counter 4 . .;set_tail_cleanup;set_pattern_sequence [A];clear_patterns;add_subpattern I 500 Injector dump;"

# Should I write to file?  I think so?  But that means then I should
# detect if I am correctly in I1D or B2D right? otherwise it makes no sense.


class Target:
    COMMAND_ADDRESS = "XFEL_SIM.UTIL/BUNCH_PATTERN/CONTROL/START_COMMAND_SEQUENCE"
    BP_ADDRESS = "XFEL_SIM.UTIL/BUNCH_PATTERN/CONTROL/PATTERN_SERIALIZATION"

    def __init__(self, di):
        self.di = di

    def save_bunch_pattern(self):
        pass

    def to_i1d(self):
        pass

    def to_b2d(self):
        pass

    def restore(self):
        # If I do I1D and then B2D, it should restore to what came before I1D, not to I1D itself!
        pass
