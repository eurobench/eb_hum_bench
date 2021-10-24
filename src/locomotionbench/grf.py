#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grf.py:
Calculates the Ground reaction forces
"""
__author__ = ["Adrià Roig"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Felix Aller", "Monika Harant", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np


class GroundReactionForces(PerformanceIndicator):

    _pi_name = 'ground_reaction_forces'
    _required = ['phases', 'ftl', 'ftr']

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
        grf, grf_agg = self.run_pi()

        if len(grf) == len(self.lead_time):
            for key in grf_agg:
                self.export_vector(grf_agg[key], f"{self.pi_name}_aggregated_{key}{self.export_file_type}")
            return 0
        else:
            return -1

    def run_pi(self):
        grf = [self.__metric(phases) for phases in zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy())]
        grf_l2 = np.linalg.norm(grf, axis=1)
        grf_agg = {'all': self.average(grf_l2)}
        for key in self.robot.step_list:
            grf_avg = self.average(grf_l2, self.robot.step_list[key])
            grf_agg[key] = self.aggregate(grf_avg)
        return grf, grf_agg

    def __metric(self, phases):
        grf_constraints = np.zeros(4)
        left, right, double, fl_obj, fr_obj = phases[0]
        if double:
            grf_constraints[0] = (abs(fl_obj.forces[0]) / fl_obj.forces[2]) + (abs(fr_obj.forces[0]) / fr_obj.forces[2])
            grf_constraints[1] = (abs(fl_obj.forces[1]) / fl_obj.forces[2]) + (abs(fr_obj.forces[1]) / fr_obj.forces[2])
            grf_constraints[2] = (abs(fl_obj.moment[0]) / fl_obj.forces[2]) + (abs(fr_obj.moment[0]) / fr_obj.forces[2])
            grf_constraints[3] = (abs(fl_obj.moment[1]) / fl_obj.forces[2]) + (abs(fr_obj.moment[1]) / fr_obj.forces[2])
        elif left:
            grf_constraints[0] = (abs(fl_obj.forces[0]) / fl_obj.forces[2])
            grf_constraints[1] = (abs(fl_obj.forces[1]) / fl_obj.forces[2])
            grf_constraints[2] = (abs(fl_obj.moment[0]) / fl_obj.forces[2])
            grf_constraints[3] = (abs(fl_obj.moment[1]) / fl_obj.forces[2])
        elif right:
            grf_constraints[0] = (abs(fr_obj.forces[0]) / fr_obj.forces[2])
            grf_constraints[1] = (abs(fr_obj.forces[1]) / fr_obj.forces[2])
            grf_constraints[2] = (abs(fr_obj.moment[0]) / fr_obj.forces[2])
            grf_constraints[3] = (abs(fr_obj.moment[1]) / fr_obj.forces[2])
        return grf_constraints
