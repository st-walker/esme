import pickle
from pathlib import Path
from random import random
from typing import Optional, Any
import fnmatch
import logging

import numpy as np

from esme.control.dint import DOOCSInterfaceABC, DOOCSAddress

LOG = logging.getLogger(__name__)

class DictionaryDOOCSInterface(DOOCSInterfaceABC):
    def __init__(self, initial_state: Optional[dict] = None):
        self._machine_state = {}
        if initial_state is not None:
            self._machine_state |= initial_state

    def get_value(self, channel: str) -> Any:
        if "*" in channel:
            return self._get_wildcard(channel)

        try:
            value = self._machine_state[channel]
        except:
            import ipdb; ipdb.set_trace()
        try:
            return value.get_value(self)
        except AttributeError:
            return value

    def _get_wildcard(self, wcard_address):
        wcard = self._machine_state[wcard_address]
        matching_columns = fnmatch.filter(wcard.snapshots_db, wcard_address)
        series = wcard.snapshots_db[matching_columns].iloc[0]
        values = series.values
        full_addresses = list(series.keys())
        locations = [DOOCSAddress.from_string(a).location for a in full_addresses]

        out_list = []
        for value, location in zip(values, locations):
            out_list.append([0, value, 0, 1_000_000, location])

        out_array = np.array(out_list)
        return out_array

    def set_value(self, channel: str, val: Any) -> None:
        LOG.debug(f"Setting, {channel} = {val}")
        self._machine_state[channel] = val

    def get_charge(self) -> float:
        return 250e-12  # ?

    # def register_channe

class ReadBackAddress:
    def __init__(self, rb, noise):
        self.rb = rb
        self.noise = noise

    def get_value(self, machine):
        value = machine.get_value(self.rb)
        value += self.noise * random()
        return value


class Alias:
    def __init__(self, alternative_name):
        self.alternative_name = alternative_name

    def get_value(self, machine):
        return machine.get_value(alternative_name)


class ReadOnlyDummyAddress:
    def __init__(self, address):
        self.address = address

    def get_value(self, machine):
        return np.nan

class ReadOnlyWildcardDummyAddress:
    def __init__(self, wildcard):
        self.wildcard = wildcard

    def get_value(self, machine):
        from IPython import embed; embed()


class WildcardAddress:
    def __init__(self, address, snapshots_db):
        self.address = address
        self.snapshots_db = snapshots_db

    def get_value(self, machine):
        pass


class QualifiedImageAddress:
    BEAM_ALLOWED_ADDRESS = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"
    def __init__(self, image_address, filters, snapshots_db, images_dir):
        self.image_address = image_address
        self.filters = filters
        self.snapshots_db = snapshots_db.reset_index().drop(columns="index")
        self.images_dir = images_dir

    def is_beam_off(self, machine):
        return not bool(machine.get_value(self.BEAM_ALLOWED_ADDRESS))

    def _get_beam_image_mask(self, machine):
        mask = np.ones((len(self.snapshots_db),), dtype=bool)

        mask = self.snapshots_db[self.BEAM_ALLOWED_ADDRESS] == 1
        

        for address in self.filters:
            machine_value = machine.get_value(address)

            # Special as it defines if a background or not
            if "QI.52.I1" in address: # Always different for some reason??
                continue

            if "QI.60.I1" in address:
                continue

            # if "QI.61.I1" in address:
            #     import ipdb; ipdb.set_trace()

            # This atol value is very important...  I am comparing
            # quadrupole strenghts, setpoints, and maybe even
            # booleans..  if i compare booleans with atol=1 then i have a prolbem!
            snapshots_db_mask = np.isclose(self.snapshots_db[address], machine_value, atol=1)
            if sum(snapshots_db_mask & mask) == 0:
                import ipdb; ipdb.set_trace()
                import sys
                sys.exit()

            mask &= snapshots_db_mask

        return mask

    def _get_background_image_mask(self, machine):
        mask = np.ones((len(self.snapshots_db),), dtype=bool)
        machine_value = machine.get_value(self.BEAM_ALLOWED_ADDRESS)
        mask = self.snapshots_db[self.BEAM_ALLOWED_ADDRESS] == machine_value
        return mask

    def get_value(self, machine):
        if self.is_beam_off(machine):
            mask = self._get_background_image_mask(machine)
        else:
            mask = self._get_beam_image_mask(machine)

            
        subdf = self.snapshots_db[mask]
        picked_index = subdf.sort_values("timestamp").index[0]
        picked_series = subdf.loc[picked_index]
        image_path = Path(picked_series[self.image_address])

        with (self.images_dir / image_path).with_suffix(".pcl").open("rb") as f:
            image = pickle.load(f)
        self.snapshots_db = self.snapshots_db.drop(index=picked_index)
        return image



class ScanMachineInterface(DictionaryDOOCSInterface):
    pass
    # def __init__(self, *args, **kwargs):
    #     super().__init__(args, kwargs)
    #     self.image_db = pd.read_pickle("/Users/stuartwalker/repos/esme/esme/control/snapshots2021.pcl")
