#!/usr/bin/env python3

import configparser
import os
from pathlib import Path

from esme.analysis import DispersionScan, TDSScan


def load_ini(fname: os.PathLike):
    config = configparser.RawConfigParser()
    files = config.read(fname)

    try:
        basepath = Path(config["common"]["basepath"])
    except KeyError:
        basepath = Path(".")


    if not files:  # If files is not empty then it went well
        raise FileNotFoundError(f"{fname} not found")

    dscan_files = [basepath / f for f in config["dscan"].values()]
    tdsscan_files = [basepath / f for f in config["tdsscan"].values()]

    return DispersionScan(dscan_files), TDSScan(tdsscan_files)
