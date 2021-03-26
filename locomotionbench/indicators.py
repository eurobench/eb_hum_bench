#!/usr/bin/env python3
"""
@package locomotionbench
@file indicators.py
@author Felix Aller, Monika Harant, Adria Roig
@brief aggregate metrics (from literature) dataframe to calculate performance indicators for eurobench
Copyright (C) 2020 Felix Aller
Distributed under the  BSD-2-Clause License.
"""

import numpy as np


def normalize(a_, time_, values_):
    time_min_ = min(time_)
    time_max_ = max(time_) - time_min_
    time_norm = [(time_point - time_min_) / time_max_ for time_point in time_]
    return np.interp(a_, time_norm, values_)


class Indicators:
    def __init__(self, exp_, metrics_, gait_phases_):
        #  reset index if double support and or halfsteps were cropped
        self.metrics = metrics_.reset_index(drop=True)
        idx = metrics_.index.values.tolist()
        #  reset index if double support and or halfsteps were cropped
        self.gait_phases = gait_phases_[min(idx): max(idx) + 1].reset_index(drop=True)
        self.requested_indicators = exp_['indicators']
        self.a = np.arange(0, 1.01, 0.01)

    def get_single_phases_per_ptype(self, ptype_):
        """
        Determine all phases correlating to a single phase type fl_single, fr_single, or double phase and return a
        2d-array containing all indices of each of the single phases
        :param ptype_: fl_single, fr_single, or double for phase identification
        :return: single_phases 2d-list containing all indices wrt to a phase
        """
        phases = self.gait_phases.query(ptype_ + ' == True').index.values.tolist()
        pmax = max(phases)
        single_phases = []
        buffer = []
        prev = -999
        for i_ in phases:
            if prev + 1 != i_ and buffer or i_ == pmax:
                single_phases.append(buffer)
                buffer = []
            buffer.append(i_)
            prev = i_
        return single_phases

    def get_all_phases_per_ptype(self, ptype_):
        """
        Determine all phases correlating to a single phase type fl_single, fr_single, or double phase and return a
        list containing all indices of the phase
        :param ptype_: fl_single, fr_single, or double for phase identification
        :return: list of all indices wrt to the phase
        """
        return self.gait_phases.query(ptype_ + ' == True').index.values.tolist()
