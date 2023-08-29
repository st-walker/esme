from datetime import datetime
from typing import Optional
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from esme.control.mint import XFELMachineInterface, DOOCSAddress

import logging

LOG = logging.getLogger(__name__)

LOG.setLevel(logging.DEBUG)


@dataclass
class SnapshotRequest:
    addresses: list[str]
    wildcards: list[str]
    image: str


class SnapshotAccumulator:
    def __init__(self, snapshotter, filename):
        self.filename = Path(filename)
        self.records = []
        self.snapshotter = snapshotter

    @property
    def outdir(self):
        return Path(self.filename).parent        

    @property
    def image_outdir(self):
        return self.outdir / "images"

    def __enter__(self):
        LOG.debug(f"Entering SnapshotAccumulator, filename={self.filename}")
        self.outdir.mkdir(exist_ok=True)
        self.image_outdir.mkdir(exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exiting...")
        df = pd.DataFrame.from_records(self.records)
        df = df.rename(self.snapshotter.expand_wildcards())
        fname = self.filename.with_suffix(".pkl")
        LOG.debug(f"Writing DataFrame to {fname}")
        df.to_pickle(fname)
        
    def take_snapshot(self, image: Optional[list] = None, **kvps):
        result = self.snapshotter.snapshot(image_dir=self.image_outdir,
                                           image=image, **kvps)
        self.records.append(result)



class Snapshotter:
    def __init__(self, request, mi: Optional[XFELMachineInterface] = None):
        self.request = request
        self.mi = mi if mi else XFELMachineInterface()

    def expand_wildcards(self):
        subaddress_full_address_map = {}
        for wildcard in self.request.wildcards:
            wildcard = DOOCSAddress.from_string(wildcard)
            assert wildcard.is_wildcard_address()
            # Now read...
            reads = self.mi.get_value(str(wildcard))
            addresses_parts = reads[:, 4]

            full_addresses = [wildcard.with_location(loc) for loc in addresses_parts]
            subaddress_full_address_map.update(dict(zip(addresses_parts, full_addresses)))

        return subaddress_full_address_map

    def snapshot(self, image_dir=None, image=None, **kvps):
        print("Taking snapshot")
        result = {}
        result.update(self.read_addresses())
        result.update(self.read_wildcards())
        result.update(self.read_image(image_dir=image_dir, image=image))
        result.update(kvps)
        return result

    def read_image(self, image_dir, image=None):
        image_address = self.request.image
        if not image_address:
            return {}

        if image is None:
            image = self.mi.get_value(image_address)

        fname = make_image_filename(image_dir)
        np.savez(fname, image=image)

        return {image_address: f"{image_dir.name}/{fname.name}"}

    def read_addresses(self) -> dict:
        return {addy: self.mi.get_value(addy) for addy in self.request.addresses}

    def read_wildcards(self) -> dict:
        result = {}
        for wildcard in self.request.wildcards:
            reads = self.mi.get_value(wildcard)
            addresses = reads[:, 4]
            values = reads[:, 1]

            result.update(dict(zip(addresses, values)))
        return result

def make_image_filename(imagedir):
    filename = datetime.utcnow().strftime(f"%Y-%m-%d-%H:%M:%S:%fUTC.npz")
    return Path(imagedir) / filename
