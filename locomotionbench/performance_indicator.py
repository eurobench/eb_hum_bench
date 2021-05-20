#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
performance_indicator.py:
Template class of a perfomance indicator
"""
__author__ = "Felix Aller"
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Matthew Millard", "Martin Felis"]
__license__ = "BSD-2"
__version__ = "0.1"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from abc import ABC, abstractmethod
from sys import exit
from functools import wraps
from time import time
import numpy as np


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r: %r  took: %f sec' % (f.__name__, args, te - ts))
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

    def write_file(self, filename, file_content):
        try:
            with open(self.output_folder + filename, 'w') as file:
                file.write(file_content)

            file.close()
        except FileNotFoundError:
            print("Failed to write output to file \"{}\"".format(filename))
            raise

    @staticmethod
    def input_to_string(input_values):
        try:
            np_input = np.array(input_values)
        except:
            try:
                np_input = input_values.to_numpy()
            except:
                print("Failed to convert input to numpy array!")
                raise
        try:
            out = np.array2string(np_input, threshold=np.inf, max_line_width=np.inf, separator=', ').replace('\n', '')
        except:
            print("Failed to convert numpy_input to string!")
            raise
        return out

    def export_scalar(self, values, filename=None):
        file_out = '''type: \'scalar\'\nvalue: {}\n'''.format(values)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    def export_vector(self, values, filename=None, labels=None):
        np_vec_str = self.input_to_string(values)

        labels_out = ""

        if labels is not None:
            if len(values) != len(labels):
                raise ValueError("Labels count doesn't match vector length!")
            labels_out = '\nlabel: {}'.format(labels)

        file_out = '''type: \'vector\'{}\nvalue: {}\n'''.format(labels_out, np_vec_str)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    def export_matrix(self, values, filename=None, row_labels=None, col_labels=None):

        np_vec_str = self.input_to_string(values)

        row_labels_out = ""
        col_labels_out = ""

        if row_labels is not None:
            row_labels_out = '\nrow_label: {}'.format(row_labels)

        if col_labels is not None:
            col_labels_out = '\ncol_label: {}'.format(col_labels)

        file_out = '''type: \'matrix\'{}{}\nvalue: {}\n'''.format(row_labels_out, col_labels_out, np_vec_str)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out
