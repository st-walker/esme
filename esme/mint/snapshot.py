# -*- coding: utf-8 -*-

"""
S.Tomin

 machine snapshot
"""
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Snapshot:
    def __init__(self):
        self.sase_sections = ["SA1", "SA2", "SA3"]
        # or list of magnet prefix to check if they are vary
        self.magnet_prefix = []  # ["QA.", "CBY.", "CAX.", "Q.", "CY.", "CX.", "CIX.", "CIY.", "QI.", "BL."]

        self.orbit_sections = {}
        self.magnet_sections = {}
        self.phase_shifter_sections = {}
        self.undulators = {}
        self.channels = []
        self.channels_tol = []

        # alarm channels
        self.alarm_channels = []
        self.alarm_bounds = []
        self.channels_track = []

        # multidim data
        self.images = []
        self.image_folders = []

    def add_alarm_channels(self, ch, min, max):
        """
        return scalar channels
        """
        self.alarm_channels.append(ch)
        self.alarm_bounds.append([min, max])

    def add_image(self, ch, folder):
        # check if folder exists and create if not
        if not os.path.exists(folder):
            os.makedirs(folder)
        if ch in self.images:
            print("WARNING: image channel is already added")
            return
        self.images.append(ch)
        self.image_folders.append(folder)

    def add_orbit_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.orbit_sections:
            print("WARNING: channel is already added")
            return
        self.orbit_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_magnet_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.magnet_sections:
            print("WARNING: channel is already added")
            return
        self.magnet_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_phase_shifter_section(self, sec_id, tol=0.001, track=True):
        if sec_id in self.phase_shifter_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.phase_shifter_sections[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_undulator(self, sec_id, tol=0.001, track=True):
        if sec_id in self.sase_sections:
            print("WARNING: channel is already added")
            return
        if sec_id in self.sase_sections:
            self.undulators[sec_id] = {"id": sec_id, "tol": tol, "track": track}

    def add_channel(self, channel, tol=None, track=True):
        if channel in self.channels:
            print("WARNING: channel is already added")
            return
        self.channels.append(channel)
        self.channels_tol.append(tol)
        self.channels_track.append(track)
