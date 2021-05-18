from locomotionbench.performance_indicator import PerformanceIndicator
import math
import numpy as np
import rbdl


class Cop(PerformanceIndicator):
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

        self.cos = self.robot.cos.to_numpy()
        self.phases = self.robot.phases
        self.len = len(self.experiment.lead_time)

    def performance_indicator(self):
        self.result = self.run_pi()
        if len(self.result) == self.len:
            return 0
        else:
            return -1

    def run_pi(self):
        result = []
        for phase in self.phases.itertuples():
            p_1, f_1, m_1, p_2, f_2, m_2 = None, None, None, None, None, None
            if phase.double:
                p_1 = phase.fl_obj.cos
                f_1 = phase.fl_obj.forces
                m_1 = phase.fl_obj.moment
                p_2 = phase.fr_obj.cos
                f_2 = phase.fr_obj.forces
                m_2 = phase.fr_obj.moment
            elif phase.fl_single:
                p_1 = phase.fl_obj.cos
                f_1 = phase.fl_obj.forces
                m_1 = phase.fl_obj.moment
            elif phase.fr_single:
                p_1 = phase.fr_obj.cos
                f_1 = phase.fr_obj.forces
                m_1 = phase.fr_obj.moment
            result.append(self.cop(p_1, f_1, m_1, p_2, f_2, m_2))
        return result

    @staticmethod
    def cop(p_1, f_1, m_1, p_2, f_2, m_2):
        cop = np.zeros(3)
        # single support
        if p_2 is None and f_2 is None and m_2 is None:
            cop[0] = - (m_1[1] / f_1[2]) + p_1[0]
            cop[1] = p_1[1] + (m_1[0] / f_1[2])
            cop[2] = p_1[2]
        # double support
        else:
            cop[0] = (-m_1[1] - m_2[1] + (f_1[2] * p_1[0]) + (f_2[2] * p_2[0])) / (f_1[2] + f_2[2])
            cop[1] = (m_1[0] + m_2[0] + (f_1[2] * p_1[1]) + (f_2[2] * p_1[1])) / (f_1[2] + f_2[2])
            cop[2] = p_2[2]
        return cop
