from locomotionbench.performance_indicator import PerformanceIndicator
import pandas as pd
import numpy as np
import rbdl
from numba import jit


class Cap(PerformanceIndicator):
    _arg_len = 2
    _pi_name = 'CoP'

    @property
    def arg_len(self):
        return self._arg_len

    @property
    def pi_name(self):
        return self._pi_name

    def __init__(self, require_, output_folder_path_, robot_, experiment_):
        super().__init__(require_, output_folder_path_, robot_, experiment_)

        pos = self.experiment.files[self.require[0]]
        vel = self.experiment.files[self.require[1]]

        self.q = pos[self.experiment.col_names].to_numpy()
        self.qdot = vel[self.experiment.col_names].to_numpy()
        self.cos = self.robot.cos.to_numpy()

        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.len = len(self.experiment.lead_time)

    def performance_indicator(self):
        self.result = self.run_pi()
        if len(self.result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.fpe(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos))
                   for q, qdot, cos in zip(self.q, self.qdot, self.cos)
                   ]
        return result

    def fpe(self, q_, qdot_, cos_):
        fpe_output = rbdl.FootPlacementEstimatorInfo()
        self.balance_tk.CalculateFootPlacementEstimator(self.robot.model, q_, qdot_, cos_, np.array([0., 0., 1.]),
                                                        fpe_output, self.omega_small, False, False)
        return fpe_output.r0F0
