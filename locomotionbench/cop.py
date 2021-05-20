from locomotionbench.performance_indicator import *
import numpy as np
import pandas as pd


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
        result = [self.__metric(row) for row in zip(self.phases[['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj']].to_numpy())]
        return result

    @staticmethod
    def __metric(row):
        left, right, double, fl_obj, fr_obj = row[0]
        p_1, f_1, m_1, p_2, f_2, m_2 = None, None, None, None, None, None
        if double:
            p_1 = fl_obj.cos
            f_1 = fl_obj.forces
            m_1 = fl_obj.moment
            p_2 = fr_obj.cos
            f_2 = fr_obj.forces
            m_2 = fr_obj.moment
        elif left:
            p_1 = fl_obj.cos
            f_1 = fl_obj.forces
            m_1 = fl_obj.moment
        elif right:
            p_1 = fr_obj.cos
            f_1 = fr_obj.forces
            m_1 = fr_obj.moment
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
