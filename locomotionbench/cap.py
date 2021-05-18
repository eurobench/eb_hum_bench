from locomotionbench.performance_indicator import PerformanceIndicator
import math
import numpy as np
import rbdl


class Cap(PerformanceIndicator):
    _arg_len = 2
    _pi_name = 'CAP'

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
        result = [self.cap(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos))
                   for q, qdot, cos in zip(self.q, self.qdot, self.cos)
                   ]
        return result

    def cap(self, q_, qdot_, cos_):
        fpe_output = rbdl.FootPlacementEstimatorInfo()
        self.balance_tk.CalculateFootPlacementEstimator(self.robot.model, q_, qdot_, cos_, np.array([0., 0., 1.]),
                                                        fpe_output, self.omega_small, False, False)
        # u_, h_, v_, gcom_
        # fpe_output.u, fpe_output.h, fpe_output.v0C0u, fpe_output.r0P0

        cap = fpe_output.r0P0 + np.multiply(fpe_output.v0C0u * math.sqrt(fpe_output.h / abs(self.robot.gravity[2])), fpe_output.u)
        return cap
