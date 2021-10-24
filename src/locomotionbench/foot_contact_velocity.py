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

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np


class FootContactVelocity(PerformanceIndicator):

    _pi_name = 'foot_contact_velocity'
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
        angular, linear, angular_agg, linear_agg= self.run_pi()
        if len(angular) == self.len and len(linear) == self.len:
            for key in angular_agg:
                self.export_vector(angular_agg[key], f"{self.pi_name}_aggregated_angular_velocity_{key}{self.export_file_type}")
            for key in linear_agg:
                self.export_vector(linear_agg[key], f"{self.pi_name}_aggregated_linear_velocity_{key}{self.export_file_type}")
            return 0
        else:
            return -1

    def run_pi(self):
        angular, linear = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), phases)
            for q, qdot, phases in
            zip(self.q, self.qdot, zip(self.phases[['fl_single', 'fr_single', 'fl_double', 'fr_double', 'fl_obj', 'fr_obj']].to_numpy()))
        ])

        angular_agg = {}
        linear_agg = {}
        angular_l2 = np.linalg.norm(angular, axis=1)
        linear_l2 = np.linalg.norm(angular, axis=1)

        for key in self.robot.step_list:
            angular_avg = self.average(angular_l2, self.robot.step_list[key])
            linear_avg = self.average(linear_l2, self.robot.step_list[key])
            angular_agg[key] = self.aggregate(angular_avg)
            linear_agg[key] = self.aggregate(linear_avg)

        return angular, linear, angular_agg, linear_agg

    def __metric(self, q_, qdot_, phases):
        fl_single, fr_single, fl_double, fr_double, fl_obj, fr_obj = phases[0]
        omega_v = 0
        if fl_single or fl_double:
            omega_v = fl_obj.get_omega_v(self.robot.model, q_, qdot_)
        elif fr_single or fr_double:
            omega_v = fr_obj.get_omega_v(self.robot.model, q_, qdot_)
        return omega_v[:3], omega_v[3:]
