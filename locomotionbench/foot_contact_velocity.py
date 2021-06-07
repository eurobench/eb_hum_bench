#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
foot_contact_velocity.py:
Calculates the angular and linear velocity of the foot contact.
"""
__author__ = ["Felix Aller", "Adrià Roig"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = "Martin Felis"
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"
from locomotionbench.performance_indicator import *
import numpy as np


class FootContactVelocity(PerformanceIndicator):

    _pi_name = 'Foot Contact Velocity'
    _required = ['pos', 'vel', 'phases']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.p_prev = 'D'
        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        result1, result2 = self.run_pi()
        if len(result1) == self.len and len(result2) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        angular, linear = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), phases)
            for q, qdot, phases in
            zip(self.q, self.qdot, zip(self.phases[['fl_single', 'fr_single', 'fl_double', 'fr_double', 'fl_obj', 'fr_obj']].to_numpy()))
        ])

        # average_angular = {}
        #
        # for key in self.robot.step_list:
        #
        #     average_angular[key] = self.aggregate(angular, self.robot.step_list[key])

        return np.array(angular), np.array(linear)

    def __metric(self, q_, qdot_, phases):
        fl_single, fr_single, fl_double, fr_double, fl_obj, fr_obj = phases[0]
        omega_v = 0
        if fl_single or fl_double:
            omega_v = fl_obj.get_omega_v(self.robot.model, q_, qdot_)
        elif fr_single or fr_double:
            omega_v = fr_obj.get_omega_v(self.robot.model, q_, qdot_)
        return omega_v[:3], omega_v[3:]
