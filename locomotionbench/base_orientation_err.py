from locomotionbench.performance_indicator import *
import numpy as np
import rbdl
import sophus as sp
from scipy.spatial.transform import Rotation as Rt


class BaseOrientationError(PerformanceIndicator):
    _arg_len = 2
    _pi_name = 'Base Orientation Error'

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
