#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
com.py:
Calculates the center of mass and corresponding performance indicators.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Matthew Millard", "Martin Felis"]
__license__ = "BSD-2"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class Com(PerformanceIndicator):
    _arg_len = 3
    _pi_name = 'CoM'

    @property
    def arg_len(self):
        return self._arg_len

    @property
    def pi_name(self):
        return self._pi_name

    def __init__(self, require_, output_folder_path_, robot_, experiment_):
        super().__init__(require_, output_folder_path_, robot_, experiment_)
        require_ = require_
        self.read_data(require_, robot_)
        self.read_data(require_, experiment_)

        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.len = len(self.experiment.lead_time)
        self.r_c, self.v_c, self.a_c, self.h_c = [], [], [], []

    @timing
    def performance_indicator(self, pi=None):
        self.run_pi()

    def loc(self):
        result = np.array(self.r_c)
        if len(result) == self.len:
            return 0
        else:
            return -1

    def vel(self):
        result = np.array(self.h_c)
        if len(result) == self.len:
            return 0
        else:
            return -1

    def acc(self):
        result = np.array(self.a_c)
        if len(result) == self.len:
            return 0
        else:
            return -1

    def ang_mom(self):
        result = []
        for item in self.h_c:
            result.append(item / (self.robot.mass * self.robot.leg_length ** 2))
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(qddot))
                  for q, qdot, qddot in zip(self.q, self.qdot, self.qddot)
                  ]
        return result

    def __metric(self, q_, qdot_, qddot_):
        r_c_ = np.zeros(3)
        v_c_ = np.zeros(3)
        a_c_ = np.zeros(3)
        h_c_ = np.zeros(3)
        model_mass_ = rbdl.CalcCenterOfMass(self.robot.model, q_, qdot_, r_c_, qddot_, v_c_, a_c_, h_c_, None, True)
        self.r_c.append(r_c_)
        self.v_c.append(v_c_)
        self.a_c.append(a_c_)
        self.h_c.append(h_c_)
        return r_c_, h_c_, v_c_, a_c_, model_mass_
