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

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np


class Cop(PerformanceIndicator):

    _pi_name = 'center_of_pressure'
    _required = ['phases', 'pos', 'cos', 'ftl', 'ftr']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)

    @timing
    def performance_indicator(self):
        trajectory, distances, integrals_agg, percentage_agg, min_max_agg, average_dist_agg = self.run_pi()
        if len(distances) == len(self.lead_time):
            for key in integrals_agg:
                self.export_vector(integrals_agg[key], f"{self.pi_name}_integrate_bos_dist_{key}{self.export_file_type}")
            for key in percentage_agg:
                self.export_vector(percentage_agg[key], f"{self.pi_name}_percentage_bos_dist_{key}{self.export_file_type}")
            for key in min_max_agg:
                self.export_vector(min_max_agg[key], f"{self.pi_name}_min_max_bos_dist_{key}{self.export_file_type}")
            for key in average_dist_agg:
                self.export_vector(average_dist_agg[key], f"{self.pi_name}_average_bos_dist_{key}{self.export_file_type}")
            return 0
        else:
            return -1

    def run_pi(self):
        trajectory, distances = zip(*[self.__metric(np.ascontiguousarray(q), row) for q, row in zip(self.q, zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy()))])
        distances = np.array(distances)

        integrals_agg = {'all': self.integrate(distances)}
        percentage_agg = {'all': self.percentage(distances)}
        min_max_agg = {'all': self.min_max(distances)}
        average_dist_agg = {'all': self.average(distances)}

        for key in self.robot.step_list:
            average_dist = self.average(distances, self.robot.step_list[key])
            percentage_pos, percentage_neg = self.percentage(distances, self.robot.step_list[key])
            min_dist, max_dist = self.min_max(distances, self.robot.step_list[key])
            integral = self.integrate(distances, self.robot.step_list[key])

            integrals_agg[key] = self.aggregate(integral)
            average_dist_agg[key] = self.aggregate(average_dist)
            percentage_agg[key] = [self.aggregate(percentage_pos), self.aggregate(percentage_neg)]
            min_max_agg[key] = [self.aggregate(min_dist), self.aggregate(max_dist)]

        return trajectory, distances, integrals_agg, percentage_agg, min_max_agg, average_dist_agg

    def __metric(self, q, row):
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

        foot1 = fl_obj
        foot2 = fr_obj
        if not fl_obj.id:  # foot1 should always be the contact foot in single support
            foot1, foot2 = foot2, foot1  # swap variables to achieve this
        cop_bos = self.robot.distance_to_support_polygon(q, cop, foot1, foot2)

        return cop, cop_bos
