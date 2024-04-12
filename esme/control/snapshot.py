from __future__ import annotations

from datetime import datetime
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
import pytz
from typing import Any, Self

import numpy as np
import pandas as pd

from esme.control.dint import DOOCSInterface, DOOCSAddress

import logging

LOG = logging.getLogger(__name__)

LOG.setLevel(logging.DEBUG)


@dataclass
class SnapshotRequest:
    addresses: list[str]
    wildcards: list[str]
    image: Optional[str]


class SnapshotAccumulator:
    def __init__(self, snapshotter: Snapshotter, filename | None : str = None):
        self.filename = Path(filename)
        self.records = []
        self.snapshotter = snapshotter

    @property
    def outdir(self) -> Path:
        return Path(self.filename).parent

    @property
    def image_outdir(self) -> Path:
        return self.outdir / "images"

    def __enter__(self) -> Self:
        LOG.debug(f"Entering SnapshotAccumulator, filename={self.filename}")
        self.outdir.mkdir(exist_ok=True)
        self.image_outdir.mkdir(exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        df = pd.DataFrame.from_records(self.records)
        df = df.rename(self.snapshotter.expand_wildcards())

        if self.filename:
            fname = self.filename.with_suffix(".pkl")
            LOG.debug(f"Writing DataFrame to {fname}")
            df.to_pickle(fname)

    def take_snapshot(self, image: Optional[list] = None, **kvps):
        result = self.snapshotter.snapshot(image_dir=self.image_outdir,
                                           image=image, **kvps)
        self.records.append(result)



class Snapshotter:
    def __init__(self, request: SnapshotRequest, di: DOOCSInterface | None = None):
        self.request = request
        self.di = di if di else DOOCSInterface()

    def expand_wildcards(self):
        subaddress_full_address_map = {}
        for wildcard in self.request.wildcards:
            wildcard = DOOCSAddress.from_string(wildcard)
            assert wildcard.is_wildcard_address()
            # Now read...
            reads = self.di.get_value(str(wildcard))
            addresses_parts = np.array(reads)[:, 4]

            full_addresses = [wildcard.with_location(loc) for loc in addresses_parts]
            subaddress_full_address_map.update(dict(zip(addresses_parts, full_addresses)))

        return subaddress_full_address_map

    def snapshot(self, image_dir: str | None = None, image: np.ndarray | None = None, **kvps):
        result = {}
        result.update(self.read_addresses())
        result.update(self.read_wildcards())
        if self.request.image:
            result.update(self.read_image(image_dir=image_dir, image=image))
        result.update(kvps)
        return result

    def read_image(self, image_dir: str, image: np.ndarray | None = None) -> dict[str, str]:
        image_address = self.request.image
        if not image_address:
            return {}

        if image is None:
            image = self.di.get_value(image_address)

        fname = make_image_filename(image_dir)
        np.savez(fname, image=image)

        return {image_address: f"{image_dir.name}/{fname.name}"}, image

    def try_and_read(self, addy: str, default: float = np.nan) -> Any:
        try:
            return self.di.get_value(addy)
        except:
            return default

    def read_addresses(self) -> dict[str, Any]:
        return {addy: self.try_and_read(addy) for addy in self.request.addresses}

    def read_wildcards(self) -> dict:
        result = {}
        for wildcard in self.request.wildcards:
            reads = np.array(self.di.get_value(wildcard))
            addresses = reads[:, 4]
            values = reads[:, 1]

            result.update(dict(zip(addresses, values)))
        return result

def make_image_filename(imagedir: str) -> Path:
    # this is the local time.
    tz = pytz.timezone("Europe/Berlin")
    local_hamburg_time = tz.localize(datetime.now())
    filename = local_hamburg_time.strftime(f"%Y-%m-%d-%H:%M:%S:%f.npz")
    return Path(imagedir) / filename
