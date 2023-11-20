import numpy as np

try:
    import pydoocs
except ImportError:
    pass
from dataclasses import dataclass

from .dint import DOOCSInterface


def get_bunch_pattern():
    return pydoocs.read("XFEL.DIAG/TIMER/DI1914TL/BUNCH_PATTERN")["data"]


def rf_transition_entries_mask(bp):
    return is_bit_set(bp, 24)


def get_any_injector_laser(bp):
    return is_bit_set(bp, 4) | is_bit_set(bp, 5) | is_bit_set(bp, 6) | is_bit_set(bp, 7)


def nbeam_regions(bp):
    transition_entries = rf_transition_entries_mask(bp)
    # Add one because we always have at least one, then we check
    # where bit 24 goes from being set to not set.
    return (np.diff(transition_entries) == -1).sum() + 1


def get_beam_region_start_stop_indices(bp):
    pairs = list(
        zip(_get_beam_region_start_indices(bp), _get_beam_region_stop_indices(bp))
    )
    return pairs


def get_transition_region_start_stop_indices(bp):
    br_pairs = iter(get_beam_region_start_stop_indices(bp))
    beam_region_stop = next(br_pairs)[1]
    result = []
    for pair in br_pairs:
        transition_region_start = beam_region_stop
        transition_region_stop = pair[0]
        beam_region_stop = pair[1]

        result.append((transition_region_start, transition_region_stop))

    result.append((beam_region_stop, len(bp)))

    return result


def _get_beam_region_start_indices(bp):
    indices = [0]
    rf_transitions = rf_transition_entries_mask(bp)
    (transition_indices,) = np.where(np.diff(rf_transitions) == -1)
    transition_indices += 1
    indices.extend(transition_indices)
    return indices


def _get_beam_region_stop_indices(bp):
    rf_transitions = rf_transition_entries_mask(bp)
    transition_indices, = np.where(np.diff(rf_transitions) == 1)
    transition_indices += 1
    return transition_indices


def nbunches_between_two_indices(bp, index0, index1):
    subpattern = bp[index0:index1]
    nbunches = any_injector_laser(bp).sum()
    return nbunches


def get_beam_regions(bp):
    pairs = get_beam_region_start_stop_indices(bp)
    result = []
    for beam_region_number, (start, stop) in enumerate(pairs, start=1):
        subpattern = bp[start:stop]
        result.append(BeamRegion(subpattern, start, beam_region_number))

    return result

def get_transition_regions(bp):
    pairs = get_transition_region_start_stop_indices(bp)
    result = []
    for start, stop in pairs:
        subpattern = bp[start:stop]
        result.append(TransitionRegion(subpattern, start))

    return result


class Section:
    T0 = 800  # first bunch pattern starts at 800 us.
    FREQUENCY = 9.028e6  # MHz
    PERIOD = 1e6 / FREQUENCY  # Microseconds

    def __len__(self):
        return len(self.subpattern)

    @property
    def tstart(self):
        return bp_index_to_us(self.istart)

    def get_indices(self):
        indices = self.istart + np.arange(len(self))
        return indices

    def nbunches(self):
        return any_injector_laser(self.subpattern).sum()

    def get_times(self):
        return bp_index_to_us(self.get_indices())


class BeamRegion(Section):
    def __init__(self, subpattern: np.ndarray, istart: int, idn: int):
        self.subpattern = subpattern
        self.istart = istart
        self.idn = idn # Beam Region Number

    def get_rep_rate_bunch_mask(self):
        pass

class TransitionRegion(Section):
    def __init__(self, subpattern: np.ndarray, istart: int):
        self.subpattern = subpattern
        self.istart = istart


def any_injector_laser(bp):
    return is_bit_set(bp, 4) | is_bit_set(bp, 5) | is_bit_set(bp, 6) | is_bit_set(bp, 7)


def is_bit_set(array, n):
    return (array >> n) & 1


def bp_index_to_us(bp_index, offset=800):
    frequency = 9.028e6
    period = 1 / frequency
    period *= 1e6  # to us

    return bp_index * period + 800


def us_to_bp_index(time, offset=800):
    frequency = 9.028e6
    period = 1 / frequency
    period *= 1e6
    return np.round((time - offset) / period).astype(int)
