#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
com.py:
Calculates the center of mass and corresponding performance indicators.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np
import rbdl


class Com(PerformanceIndicator):
    _pi_name = 'CoM'
    _required = ['pos', 'vel', 'acc']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.r_c, self.v_c, self.a_c, self.h_c = [], [], [], []

    @timing
    def performance_indicator(self, pi=None):
        r_c, com_velocity_avg, com_acceleration_avg, h_c_avg_agg, h_c_integral_agg = self.run_pi()
        if len(r_c) == len(self.lead_time):
            return 0
        else:
            return -1

    def run_pi(self):
        #  r_c location; h_c velocity; a_c acceleration, h_c normalized ang momentum about com
        r_c, h_c, v_c, a_c = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(qddot))
            for q, qdot, qddot in zip(self.q, self.qdot, self.qddot)
            ])

        v_c_l2 = np.linalg.norm(v_c, axis=1)
        a_c_l2 = np.linalg.norm(a_c, axis=1)

        com_velocity_avg = self.average(v_c_l2)
        com_acceleration_avg = self.average(a_c_l2)
        h_c = np.abs(np.array(h_c))
        h_c_avg_agg = {'all': [self.average(h_c[0]), self.average(h_c[1]), self.average(h_c[2])]}
        h_c_integral_agg = {'all': [self.average(h_c[0]), self.average(h_c[1]), self.average(h_c[2])]}

        for key in self.robot.step_list:
            h_c_avg_x = self.average(h_c[:, 0], self.robot.step_list[key])
            h_c_avg_y = self.average(h_c[:, 1], self.robot.step_list[key])
            h_c_avg_z = self.average(h_c[:, 2], self.robot.step_list[key])
            integral_x = self.integrate(h_c[:, 0], self.robot.step_list[key])
            integral_y = self.integrate(h_c[:, 1], self.robot.step_list[key])
            integral_z = self.integrate(h_c[:, 2], self.robot.step_list[key])

            h_c_avg_agg[key] = [self.aggregate(h_c_avg_x), self.aggregate(h_c_avg_y), self.aggregate(h_c_avg_z)]
            h_c_integral_agg[key] = [self.aggregate(integral_x), self.aggregate(integral_y), self.aggregate(integral_z)]
        return r_c, com_velocity_avg, com_acceleration_avg, h_c_avg_agg, h_c_integral_agg

    def __metric(self, q_, qdot_, qddot_):
        r_c_ = np.zeros(3)
        v_c_ = np.zeros(3)
        a_c_ = np.zeros(3)
        h_c_ = np.zeros(3)
        model_mass_ = rbdl.CalcCenterOfMass(self.robot.model, q_, qdot_, r_c_, qddot_, v_c_, a_c_, h_c_, None, True)
        return r_c_, h_c_, v_c_, a_c_
