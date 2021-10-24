#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
res_forces.py:
Calculates the residual forces.
"""
__author__ = ["Felix Aller"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np
import rbdl
import sophus as sp
from scipy.spatial.transform import Rotation as Rt


class ResidualForces(PerformanceIndicator):

    _pi_name = 'residual_forces'
    _required = ['trq']

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
        trq_agg, res_forces = self.run_pi()

        if len(trq_agg) == 6:
            for key in res_forces:
                self.export_vector(res_forces[key], f"{self.pi_name}_{key}{self.export_file_type}")
            return 0
        else:
            return -1

    def run_pi(self):
        freeflyer = self.robot.base_link[0:6]
        res_forces = {}
        trq_agg = []
        for key in self.robot.step_list:
            trq_agg = []
            for col in freeflyer:
                trq_avg = self.average(abs(self.trq[col].to_numpy()), self.robot.step_list[key])
                trq_agg.append(self.aggregate(trq_avg))
            res_forces[key] = trq_agg
        return trq_agg, res_forces

