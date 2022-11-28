#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 07:24:42 2019

@author: xfeloper
"""

import sys
import time

sys.path.append("/home/xfeloper/user/tomins/ocelot_new/pyBigBro")
import time

import config_inj_study as conf
import pydoocs
from esme.mint.machine import MPS, Machine
from mint.snapshot import Snapshot, SnapshotDB

import logging

LOG = logging.getLogger(__name__)



def take_background(db, machine, nshots=5):
    print("background taking ... beam off")
    mps = MPS()
    mps.beam_off()
    time.sleep(1)
    for i in range(nshots):
        df = machine.get_machine_snapshot()
        db.add(df)
        time.sleep(1)
    print("background taking is over ... beam on")
    mps.beam_on()


def set_initial_optics():
    # High dispersion optics
    quads = [
        "XFEL.MAGNETS/MAGNET.ML/QI.52.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.53.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.54.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.55.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.57.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.59.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.60.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.61.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP",
    ]


    kk12 = [-83.7203, 500.308, 188.82, -712.48, 712.48, -712.48, -509.559,
            832.63, -249.585, 831.95]

    # set
    for i, q in enumerate(quads):
        LOG.info("setting initial optics, quad: ", q, " <--", kk12[i])
        pydoocs.write(q, kk12[i])


def set_dscan_quads(dx):
    quads = [
        "XFEL.MAGNETS/MAGNET.ML/QI.60.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.61.I1/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.63.I1D/KICK_MRAD.SP",
        "XFEL.MAGNETS/MAGNET.ML/QI.64.I1D/KICK_MRAD.SP",
    ]

    kk12 = [-509.559, 832.63, -249.585, 831.95]
    kk10 = [-509.0345, 832.175, 106.965, 475.4]
    kk08 = [-508.965, 820.2076, 582.365, 0]
    kk06 = [-508.749, 789.625, 1046.306, -475.4]

    if dx == 12:
        kk = kk12
    elif dx == 10:
        kk = kk10
    elif dx == 8:
        kk = kk08
    elif dx == 6:
        kk == kk06
    else:
        raise ValueError("Unknown dispersion setting: pick from {12, 10, 8, 6}.")

    LOG.info("Setting quads for OCR dispersion: dx={dx/10.}m")

    for i, q in enumerate(quads):
        LOG.info("setting downstream quad: ", q, " <--", kk[i])
        pydoocs.write(q, kk[i])


# def take_data(actual_dx, percentage, extra_stuff=""):

#     snapshot = conf.snapshot

#     now = time.strftime("%Y%m%d-%H_%M_%S")
#     fname = f"{now}-Dx-{dx}-{percentage}%"
#     if extra_stuff:
#         fname += "-" + extra_stuff
#     fname += ".pcl"

#     machine = Machine(snapshot)
#     db = SnapshotDB(filename=)
#     df_ref = machine.get_machine_snapshot()
#     db.add(df_ref)

#     time.sleep(1)

#     take_background(db, machine)
#     time.sleep(1)

#     for i in range(30):
#         LOG.info(f"writing image {i}")
#         df = machine.get_machine_snapshot(check_if_online=True)
#         db.add(df)
#         time.sleep(1)

#     db.save()
