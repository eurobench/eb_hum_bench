#!/usr/bin/env python
from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class Zmp(PerformanceIndicator):
    _arg_len = 3
    _pi_name = 'ZMP'

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

        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(qddot))
                  for q, qdot, qddot in zip(self.q, self.qdot, self.qddot)]
        return result

    def __metric(self, q_, qdot_, qddot_):
        zmp = np.zeros(3)
        rbdl.CalcZeroMomentPoint(self.robot.model, q_, qdot_, qddot_, zmp, np.array([0., 0., 1.]), np.array([0., 0., 1.]),
                                 True)
        return zmp
