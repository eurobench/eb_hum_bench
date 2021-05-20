#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_orientation.py:
Calculates the average floor impact.
"""
__author__ = ["Adrià Roig", "Felix Aller"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = "Monika Harant"
__license__ = "BSD-2"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *


class Impact(PerformanceIndicator):

    _pi_name = 'Impact'
    _required = ['phases']

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
        result = self.run_pi()
        if result:
            return 0
        else:
            return -1

    def run_pi(self):
        impact, no_impacts = zip(*[self.__metric(phases) for phases in zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy())])
        impact, no_impacts = sum(impact), sum(no_impacts)
        result = impact / (no_impacts * self.robot.mass)
        return result

    def __metric(self, phases):
        left, right, double, fl_obj, fr_obj = phases[0]
        if double:
            f_1, f_2 = fl_obj.forces[self.vertical_force], fr_obj.forces[self.vertical_force]
            return f_1 + f_2, 2
        elif left:
            return fl_obj.forces[self.vertical_force], 1
        elif right:
            return fr_obj.forces[self.vertical_force], 1
