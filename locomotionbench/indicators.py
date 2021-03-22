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

    def get_left_single_support_indices(self):
        pass

