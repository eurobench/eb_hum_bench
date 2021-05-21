#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cap.py:
Calculates the foot placement estimator and corresponding performance indicators.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Matthew Millard", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class Fpe(PerformanceIndicator):

    _pi_name = 'FPE'
    _required = ['pos', 'vel', 'cos', 'phases']

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
        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        trajectory, result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        trajectory, result = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos), obj)
            for q, qdot, cos, obj in
            zip(self.q, self.qdot, self.cos, zip(self.phases[['fl_obj', 'fr_obj']].to_numpy()))
        ])
        return trajectory, result

    def __metric(self, q_, qdot_, cos_, obj=None):
        fpe_output = rbdl.FootPlacementEstimatorInfo()
        self.balance_tk.CalculateFootPlacementEstimator(self.robot.model, q_, qdot_, cos_, np.array([0., 0., 1.]),
                                                        fpe_output, self.omega_small, False, True)
        if obj:
            foot1, foot2 = obj[0]
            if not foot1.id:  # foot1 should always be the contact foot in single support
                foot1, foot2 = foot2, foot1  # swap variables to achieve this
            fpe_bos = self.robot.distance_to_support_polygon(q_, fpe_output.r0F0, foot1, foot2)
            return fpe_output.r0F0, fpe_bos
        return fpe_output.r0F0
