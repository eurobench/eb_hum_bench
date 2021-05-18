from abc import ABC, abstractmethod
import yaml
from os import path
from sys import argv, exit
from datetime import datetime


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
        self.robot = robot_
        self.experiment = experiment_
        self.require = require_
        self.output_folder = output_folder_path

        # self.performance_indicator()


    @abstractmethod
    def performance_indicator(self):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def run_pi(self):
        raise NotImplementedError("Please Implement this method")


