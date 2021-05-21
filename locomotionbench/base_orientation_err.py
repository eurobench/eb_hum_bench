#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
base_orientation.py:
Calculates the base orientation error.
"""
__author__ = ["Adrià Roig", "Felix Aller"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.2"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from locomotionbench.performance_indicator import *
import numpy as np
import rbdl
import sophus as sp
from scipy.spatial.transform import Rotation as Rt


class BaseOrientationError(PerformanceIndicator):

    _pi_name = 'Base Orientation Error'
    _required = ['pos', 'phases']

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
        result = self.run_pi()

        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(np.ascontiguousarray(q)) for q in self.q]
        return result

    def __metric(self, q_):
        base_orient = rbdl.CalcBodyWorldOrientation(self.robot.model, q_, self.robot.body_map.index(self.robot.torso_link) + 1, True).transpose()
        r_base_orient = Rt.from_matrix(base_orient)
        r_identity = Rt.from_quat([0, 0, 0, 1])
        r_error = r_identity * r_base_orient.inv()
        s_error = sp.SO3(r_error.as_matrix())
        error = sp.SO3.log(s_error)
        return error
