#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cop.py:
Calculates the center of pressure and corresponding performance indicators.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np


class Cop(PerformanceIndicator):

    _pi_name = 'CoP'
    _required = ['phases', 'cos', 'ftl', 'ftr']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

        self.len = len(self.robot.cos)

    @timing
    def performance_indicator(self):
        result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(row) for row in zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy())]
        return result

    @staticmethod
    def __metric(row):
        left, right, double, fl_obj, fr_obj = row[0]
        p_1, f_1, m_1, p_2, f_2, m_2 = None, None, None, None, None, None
        if double:
            p_1 = fl_obj.cos
            f_1 = fl_obj.forces
            m_1 = fl_obj.moment
            p_2 = fr_obj.cos
            f_2 = fr_obj.forces
            m_2 = fr_obj.moment
        elif left:
            p_1 = fl_obj.cos
            f_1 = fl_obj.forces
            m_1 = fl_obj.moment
        elif right:
            p_1 = fr_obj.cos
            f_1 = fr_obj.forces
            m_1 = fr_obj.moment
        cop = np.zeros(3)
        # single support
        if p_2 is None and f_2 is None and m_2 is None:
            cop[0] = - (m_1[1] / f_1[2]) + p_1[0]
            cop[1] = p_1[1] + (m_1[0] / f_1[2])
            cop[2] = p_1[2]
        # double support
        else:
            cop[0] = (-m_1[1] - m_2[1] + (f_1[2] * p_1[0]) + (f_2[2] * p_2[0])) / (f_1[2] + f_2[2])
            cop[1] = (m_1[0] + m_2[0] + (f_1[2] * p_1[1]) + (f_2[2] * p_1[1])) / (f_1[2] + f_2[2])
            cop[2] = p_2[2]
        return cop
