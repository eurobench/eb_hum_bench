from locomotionbench.performance_indicator import *
import math
import numpy as np
import rbdl


class Cap(PerformanceIndicator):
    _arg_len = 3
    _pi_name = 'CAP'

    @property
    def arg_len(self):
        return self._arg_len

    @property
    def pi_name(self):
        return self._pi_name

    def __init__(self, require_, output_folder_path_, robot_, experiment_):
        super().__init__(require_, output_folder_path_, robot_, experiment_)
        require_ = require_
        self.read_data(require_, robot_)
        self.read_data(require_, experiment_)

        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = [self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos))
                  for q, qdot, cos in zip(self.q, self.qdot, self.cos)
                  ]
        return result

    def __metric(self, q_, qdot_, cos_):
        fpe_output = rbdl.FootPlacementEstimatorInfo()
        self.balance_tk.CalculateFootPlacementEstimator(self.robot.model, q_, qdot_, cos_, np.array([0., 0., 1.]),
                                                        fpe_output, self.omega_small, False, True)
        # u_, h_, v_, gcom_
        # fpe_output.u, fpe_output.h, fpe_output.v0C0u, fpe_output.r0P0

        cap = fpe_output.r0P0 + np.multiply(fpe_output.v0C0u * math.sqrt(fpe_output.h / abs(self.robot.gravity[2])),
                                            fpe_output.u)
        return cap
