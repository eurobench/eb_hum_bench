#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cap.py:
Calculates the zero moment point and corresponding performance indicators.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Martin Felis"]
__license__ = "BSD-2"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class Zmp(PerformanceIndicator):

    _pi_name = 'ZMP'
    _required = ['pos', 'vel', 'acc', 'phases']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(qddot))
                  for q, qdot, qddot in zip(self.q, self.qdot, self.qddot)]
        return result

    def __metric(self, q_, qdot_, qddot_):
        zmp = np.zeros(3)
        rbdl.CalcZeroMomentPoint(self.robot.model, q_, qdot_, qddot_, zmp, np.array([0., 0., 1.]), np.array([0., 0., 1.]),
                                 True)
        return zmp
