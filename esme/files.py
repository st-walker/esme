#!/usr/bin/env python3

import configparser
import os

from esme.analysis import DispersionScan, TDSScan


def load_ini(fname: os.PathLike):
    config = configparser.ConfigParser()
    files = config.read(fname)

    if not files:  # If files is not empty then it went well
        raise FileNotFoundError(f"{fname} not found")

    dscan_files = list(config["dscan"].values())
    tdsscan_files = list(config["tdsscan"].values())

    return DispersionScan(dscan_files), TDSScan(tdsscan_files)
