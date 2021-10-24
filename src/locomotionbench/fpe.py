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

from src.locomotionbench.performance_indicator import PerformanceIndicator, timing
import numpy as np
import rbdl


class Fpe(PerformanceIndicator):

    _pi_name = 'foot_placement_estimator'
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
        trajectory, distances = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos), obj)
            for q, qdot, cos, obj in
            zip(self.q, self.qdot, self.cos, zip(self.phases[['fl_obj', 'fr_obj']].to_numpy()))
        ])

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
