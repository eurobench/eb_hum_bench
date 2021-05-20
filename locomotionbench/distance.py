#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
distance.py:
Calculates the normalized travelled distance.
"""
__author__ = ["Adrià Roig", "Felix Aller"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Martin Felis"]
__license__ = "BSD-2"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class DistanceTravelled(PerformanceIndicator):

    _pi_name = 'Distance Travelled'
    _required = ['pos', 'phases']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.gait_direction = 0
        self.p_prev = np.array([0., 0., 0.])
        self.distance_traveled = 0
        self.support_type = 'DS'
        self.n_steps = 0

    @timing
    def performance_indicator(self):
        result, is_ok = self.run_pi()
        if all(is_ok):
            return 0
        else:
            return -1

    def run_pi(self):
        is_ok = [self.__metric(np.ascontiguousarray(q), phases) for q, phases in zip(self.q, zip(self.phases[['fl_single', 'fr_single', 'double']].to_numpy()))]
        result = self.distance_traveled / (self.n_steps * self.robot.leg_length)
        return result, is_ok

    def __metric(self, q_, phases):
        left, right, double = phases[0]
        if double:
            self.support_type = 'DS'
        elif left:
            if self.support_type == 'LS':
                p = rbdl.CalcBodyToBaseCoordinates(self.robot.model, q_, self.robot.body_map.index(self.robot.l_foot) + 1,
                                                   np.array([0., 0., 0.]), True)
                self.distance_traveled += p[self.gait_direction] - self.p_prev[self.gait_direction]
                self.p_prev = p
                self.n_steps += 1
            self.support_type = 'LS'
        elif right:
            if self.support_type == 'RS':
                p = rbdl.CalcBodyToBaseCoordinates(self.robot.model, q_, self.robot.body_map.index(self.robot.r_foot) + 1,
                                                   np.array([0., 0., 0.]), True)
                self.distance_traveled += p[self.gait_direction] - self.p_prev[self.gait_direction]
                self.p_prev = p
                self.n_steps += 1
            self.support_type = 'RS'
        else:
            return 0
        return 1
