#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
performance_indicator.py:
Template class of a perfomance indicator
"""
__author__ = "Felix Aller"
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Matthew Millard", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.1"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

from abc import ABC, abstractmethod
from functools import wraps
from time import time
import numpy as np


#  debugging function to use @timing decorator to obtain runtime information of individual performance indicators
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('%r: %r  took: %f sec' % (f.__name__, args, te - ts))
        return result

    return wrap


#  Template for implementing a performance indicator
class PerformanceIndicator(ABC):

    @property
    @abstractmethod
    def pi_name(self):  # name of PI; currently not used
        raise NotImplementedError()

    @pi_name.setter
    @abstractmethod
    def pi_name(self, _pi_name):
        raise NotImplementedError()

    @property
    @abstractmethod
    def required(self):  # list of required files for computation of performance indicator
        raise NotImplementedError()

    @required.setter
    @abstractmethod
    def required(self, _required):
        raise NotImplementedError()

    @abstractmethod
    def __init__(self, output_folder_path, robot=None, experiment=None):

        # set name of classes to distinguish in read_data()
        self.cl_name_robot = 'Robot'
        self.cl_name_experiment = 'Experiment'

        # set output folder path for results
        self.output_folder = output_folder_path

        # check if robot was provided
        # TODO: this is currently always the case for robot and experiment, maybe remove the None option
        if robot:
            self.robot = robot
            try:  # pass down error if files were not found in order to skip the metric calculation
                self.read_data(self.required, robot)
            except FileNotFoundError:
                raise FileNotFoundError
        if experiment:
            self.experiment = experiment
            try:  # pass down error if files were not found in order to skip the metric calculation
                self.read_data(self.required, experiment)
                self.lead_time = experiment.lead_time
            except FileNotFoundError:
                raise FileNotFoundError

    def read_data(self, require_, data_):
        name = type(data_).__name__  # are we looking at robot or experiment data
        for item in require_:
            if name == self.cl_name_experiment:  # if experiment
                # was the file provided / could be read when experiment class was created
                if item in self.experiment.files_not_provided:
                    print(f"\033[91m{item} is not provided but required.\033[0m")
                    raise FileNotFoundError  # pass down error
                if item == 'pos':
                    self.q = data_.get_file(item)[self.experiment.col_names].to_numpy()
                if item == 'vel':
                    self.qdot = data_.get_file(item)[self.experiment.col_names].to_numpy()
                if item == 'acc':
                    self.qddot = data_.get_file(item)[self.experiment.col_names].to_numpy()
                if item == 'trq':
                    self.trq = data_.get_file(item)
                if item == 'ftl':
                    self.ftl = data_.get_file(item)
                if item == 'ftr':
                    self.ftr = data_.get_file(item)
            elif name == self.cl_name_robot:  # if robot
                if item == 'cos':
                    self.cos = data_.cos.to_numpy()
                if item == 'phases':
                    self.phases = data_.phases

    def normalize(self, data):
        self.len = self.experiment.lead_time


    # implement standard call of the performance indicator from the created object
    @abstractmethod
    def performance_indicator(self):
        raise NotImplementedError("Please Implement this method")

    # implement call of actual metric and return it to performance indicator
    @abstractmethod
    def run_pi(self):
        raise NotImplementedError("Please Implement this method")

    # write file in output path
    def write_file(self, filename, file_content):
        try:
            with open(self.output_folder + filename, 'w') as file:
                file.write(file_content)

            file.close()
        except FileNotFoundError:
            print("Failed to write output to file \"{}\"".format(filename))
            raise

    # convert numerical type into string
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

    # export data to scalar
    def export_scalar(self, values, filename=None):
        file_out = '''type: \'scalar\'\nvalue: {}\n'''.format(values)

        if filename is not None:
            self.write_file(filename, file_out)

        return file_out

    # export data to vector
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

    # export data to matrix
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
