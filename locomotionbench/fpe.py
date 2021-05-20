from locomotionbench.performance_indicator import *
import numpy as np
import rbdl


class Fpe(PerformanceIndicator):
    _arg_len = 3
    _pi_name = 'FPE'

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

        self.balance_tk = rbdl.BalanceToolkit()
        self.omega_small = 1e-6
        self.len = len(self.experiment.lead_time)

    @timing
    def performance_indicator(self):
        trajectory, result = self.run_pi()
        if len(result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        trajectory, result = zip(*[
            self.__metric(np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(cos), obj)
            for q, qdot, cos, obj in
            zip(self.q, self.qdot, self.cos, zip(self.phases[['fl_obj', 'fr_obj']].to_numpy()))
        ])
        return trajectory, result

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
