#!/usr/bin/env python3
"""
@package locomotionbench
@file experiment_factory.py
@author Felix Aller
@brief take care of config file parsing and create metric objects based on different runs and trials
Copyright (C) 2020 Felix Aller
Distributed under the  BSD-2-Clause License.
"""
import pandas as pd
import numpy as np

class ExperimentFactory:
    def __init__(self, exp_, robot_):
        self.columns = ['run', 'trial', 'model_path', 'pos', 'vel', 'acc', 'trq', 'ftl', 'ftr', 'metrics']
        self.protocol = exp_['protocol']
        self.project = exp_['project']
        self.exp = exp_
        self.proj_path = exp_['inputdir'] + '/' + exp_['project'] + '/'
        self.runs = exp_['run']
        self.trial = exp_['trial']
        self.requested_indicators = exp_['indicators']
        self.model = robot_['modelpath'] + '/' + robot_['robotmodel']
        self.sep = self.exp['separator']
        self.stack_of_tasks = self.build_stack_of_tasks()
        pass

    def build_stack_of_tasks(self):
        protocol = []
        project = []
        run = []
        trial = []
        model_path = []
        pos_path = []
        vel_path = []
        acc_path = []
        trq_path = []
        ftl_path = []
        ftr_path = []
        metrics = []
        for i_ in self.runs:
            for j_ in self.trial:
                path = self.proj_path + str(i_) + '/' + str(j_) + '/'
                model_path.append(self.proj_path + self.model)
                run.append(str(i_))
                trial.append(str(j_))
                protocol.append(self.protocol)
                project.append(self.project)
                metrics.append(self.requested_indicators)
                pos_path.append(path + self.exp['files']['pos'])
                vel_path.append(path + self.exp['files']['vel'])
                acc_path.append(path + self.exp['files']['acc'])
                trq_path.append(path + self.exp['files']['trq'])
                if self.exp['files']['ftl'] and self.exp['files']['ftr']:
                    ftl_path.append(path + self.exp['files']['ftl'])
                    ftr_path.append(path + self.exp['files']['ftr'])
        stack = np.array([run, trial, model_path, pos_path, vel_path, acc_path, trq_path, ftl_path, ftr_path, metrics])
        return pd.DataFrame(columns=self.columns, data=np.transpose(stack))

    def get_stack_of_tasks(self):
        return self.stack_of_tasks

    def process_stack_of_tasks(self):
        pass
