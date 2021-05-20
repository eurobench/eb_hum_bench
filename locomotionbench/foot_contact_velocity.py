#!/usr/bin/env python3
from locomotionbench.performance_indicator import *
import numpy as np


class FootContactVelocity(PerformanceIndicator):
    _arg_len = 3
    _pi_name = 'Foot Contact Velocity'

    @property
    def arg_len(self):
        return self._arg_len

    @property
    def pi_name(self):
        return self._pi_name

    def __init__(self, require_, output_folder_path_, robot_, experiment_):
        super().__init__(require_, output_folder_path_, robot_, experiment_)

        self.read_data(require_, robot_)
        self.read_data(require_, experiment_)
        self.p_prev = 'D'
        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        result1, result2 = self.run_pi()
        if len(result1) == self.len and len(result2) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        angular, linear = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), phases)
            for q, qdot, phases in
            zip(self.q, self.qdot, zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy()))
        ])
        return np.array(angular), np.array(linear)

    def __metric(self, q_, qdot_, phases):
        left, right, double, fl_obj, fr_obj = phases[0]
        omega_v = 0
        if left:
            self.p_prev = 'L'
            omega_v = fl_obj.get_omega_v(self.robot.model, q_, qdot_)
        elif right:
            self.p_prev = 'L'
            omega_v = fr_obj.get_omega_v(self.robot.model, q_, qdot_)
        elif double:
            if self.p_prev == 'L' or self.p_prev == 'D':
                omega_v = fl_obj.get_omega_v(self.robot.model, q_, qdot_)
            if self.p_prev == 'R':
                omega_v = fr_obj.get_omega_v(self.robot.model, q_, qdot_)
        return omega_v[:3], omega_v[3:]
