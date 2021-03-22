#!/usr/bin/env python3
"""
@package locomotionbench
@file indicators.py
@author Felix Aller, Monika Harant, Adria Roig
@brief aggregate metrics (from literature) dataframe to calculate performance indicators for eurobench
Copyright (C) 2020 Felix Aller
Distributed under the  BSD-2-Clause License.
"""


class Indicators:
    def __init__(self, exp_, metrics_):
        self.metrics = metrics_.reset_index(drop=True)

        self.requested_indicators = exp_['indicators']

    def ping(self):
        pass
