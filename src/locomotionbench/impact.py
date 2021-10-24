#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_orientation.py:
Calculates the average floor impact.
"""
__author__ = ["Adrià Roig", "Felix Aller"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = "Monika Harant"
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np


class Impact(PerformanceIndicator):

    _pi_name = 'vertical_impact'
    _required = ['phases', 'ftl', 'ftr']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.vertical_force = 2

    @timing
    def performance_indicator(self):
        impact, impact_agg = self.run_pi()
        if len(impact) == len(self.lead_time):
            for key in impact_agg:
                self.export_vector(impact_agg[key], f"{self.pi_name}_maximum_aggregated_{key}{self.export_file_type}")
            return 0
        else:
            return -1

    def run_pi(self):
        impact = [self.__metric(phases) for phases in zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy())]
        max_impact_agg = {'all': self.aggregate(impact)}
        impact = np.array(impact)
        for key in self.robot.step_list:
            impact_min, impact_max = self.min_max(impact, self.robot.step_list[key])
            max_impact_agg[key] = self.aggregate(impact_max)

        return impact, max_impact_agg

    def __metric(self, phases):
        left, right, double, fl_obj, fr_obj = phases[0]
        if double:
            return fl_obj.forces[self.vertical_force] + fr_obj.forces[self.vertical_force]
        elif left:
            return fl_obj.forces[self.vertical_force]
        elif right:
            return fr_obj.forces[self.vertical_force]
