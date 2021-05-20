from abc import ABC, abstractmethod
import yaml
from os import path
from sys import argv, exit
from datetime import datetime
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return wrap


class PerformanceIndicator(ABC):

    @property
    @abstractmethod
    def arg_len(self):
        raise NotImplementedError()

    @arg_len.setter
    @abstractmethod
    def arg_len(self, _arg_len):
        raise NotImplementedError()

    @property
    @abstractmethod
    def pi_name(self):
        raise NotImplementedError()

    @pi_name.setter
    @abstractmethod
    def pi_name(self, _pi_name):
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, require_, output_folder_path, robot_=None, experiment_=None):
        if len(require_) != self.arg_len:
            print("[Performance Indicator ", self.pi_name, "] Error: List of file paths does not match required:")
            print(require_)
            exit(-1)
        self.cl_name_robot = 'Robot'
        self.cl_name_experiment = 'Experiment'
        self.robot = robot_
        self.experiment = experiment_
        self.require = require_
        self.output_folder = output_folder_path

        # self.performance_indicator()

    def read_data(self, require_, data_, d_bos=False):
        name = type(data_).__name__
        for item in require_:
            if item == 'pos' and name == self.cl_name_experiment:
                self.q = data_.files[item][self.experiment.col_names].to_numpy()
            if item == 'vel' and name == self.cl_name_experiment:
                self.qdot = data_.files[item][self.experiment.col_names].to_numpy()
            if item == 'acc' and name == self.cl_name_experiment:
                self.qddot = data_.files[item][self.experiment.col_names].to_numpy()
            if item == 'trq' and name == self.cl_name_experiment:
                self.trq = data_.files[item]
            if item == 'ftl' and name == self.cl_name_experiment:
                self.ftl = data_.files[item]
            if item == 'ftr' and name == self.cl_name_experiment:
                self.ftr = data_.files[item]
            if item == 'cos' and name == self.cl_name_robot:
                self.cos = data_.cos.to_numpy()
            if item == 'phases' and name == self.cl_name_robot or d_bos is True:
                self.phases = self.robot.phases

    # @abstractmethod
    # def verify_required(self):
    #     raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def performance_indicator(self):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def run_pi(self):
        raise NotImplementedError("Please Implement this method")
