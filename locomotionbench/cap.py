#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cap.py:
Calculates the capture point and corresponding performance indicators.
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
import math
import numpy as np
import rbdl
import pandas as pd
import matplotlib.pyplot as plt


class Cap(PerformanceIndicator):
    _pi_name = 'CAP'
    _required = ['pos', 'vel', 'cos', 'phases']

    @property
    def pi_name(self):
        return self._pi_name

    @property
    def required(self):
        return self._required

    def __init__(self, output_folder_path, robot=None, experiment=None):
        super().__init__(output_folder_path, robot, experiment)
        self.raw = pd.DataFrame(columns=['trajectory, distance'])
        self.results = pd.DataFrame(columns=['mean', 'std', 'percentage_over', 'integral', 'average'])
        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.filename = 'capture_point.yaml'

    @timing
    def performance_indicator(self):
        trajectory, distances, integrals, aggregates = self.run_pi()
        # self.__aggregate(distance)
        if len(distances) == self.lead_time:
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
        average_dist_agg = {'all': self.average_dist(distances)}

        for key in self.robot.step_list:
            average_dist = self.average_dist(distances, self.robot.step_list[key])
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
        # u_, h_, v_, gcom_
        # fpe_output.u, fpe_output.h, fpe_output.v0C0u, fpe_output.r0P0

        cap = fpe_output.r0P0 + np.multiply(fpe_output.v0C0u * math.sqrt(fpe_output.h / abs(self.robot.gravity[2])),
                                            fpe_output.u)
        if obj:
            foot1, foot2 = obj[0]
            if not foot1.id:  # foot1 should always be the contact foot in single support
                foot1, foot2 = foot2, foot1  # swap variables to achieve this
            cap_bos = self.robot.distance_to_support_polygon(q_, cap, foot1, foot2)
            return cap, cap_bos
        return cap

    def __plot(self, data):
        fig = plt.figure()
        time = np.array(self.lead_time)
        color = 'black'
        for i in range(len(self.lead_time)):
            if any(i in sl for sl in self.robot.step_list['fl_single']):
                color = 'blue'
            elif any(i in sl for sl in self.robot.step_list['fr_single']):
                color = 'red'
            elif any(i in sl for sl in self.robot.step_list['fl_double']):
                color = 'grey'
            elif any(i in sl for sl in self.robot.step_list['fr_double']):
                color = 'grey'
            else:
                continue
            plt.plot(i, data[i], color=color, marker="x")
        plt.show()
