#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gait_event_conversion.py:
Convert from eurobench gait events file to a frame-based vector of gait events and vice versa
"""
__author__ = "Felix Aller"
__copyright__ = "Copyright 2021, EUROBENCH Project"
__license__ = "BSD-2-Clause"
__version__ = "0.1"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

import sys
import numpy as np
import argparse
import pathlib
import yaml
import pandas as pd


def get_phase_dict_len(strike, toe):
    if len(strike) > len(toe):
        return len(strike)
    else:
        return len(toe)


def framelist_to_gaitevents(phase_list, lead_time):
    if type(phase_list) is dict:
        phase_list = np.array(phase_list['phases'])
    else:
        phase_list = np.array(phase_list)
    left, right, double = 'l', 'r', 'd'
    r_heel_strike, l_heel_strike, r_toe_off, l_toe_off = [], [], [], []
    index_list = np.where(phase_list[:-1] != phase_list[1:])[0] + 1
    index_list = np.insert(index_list, 0, 0)

    for index in index_list:
        if index == 0:
            if phase_list[index] == double:
                l_heel_strike.append(index)
                r_heel_strike.append(index)
            elif phase_list[index] == left:
                l_heel_strike.append(index)
            elif phase_list[index] == right:
                r_heel_strike.append(index)
        elif phase_list[index] == double:
            if phase_list[index - 1] == left:
                r_heel_strike.append(index)
            elif phase_list[index - 1] == right:
                l_heel_strike.append(index)
        elif phase_list[index] == left:
            r_toe_off.append(index)
        elif phase_list[index] == right:
            l_toe_off.append(index)

    phase_dict_timestamps = {'r_heel_strike': lead_time[r_heel_strike].flatten(), 'l_heel_strike': lead_time[l_heel_strike].flatten(), 'r_toe_off': lead_time[r_toe_off].flatten(), 'l_toe_off': lead_time[l_toe_off].flatten()}

    return phase_dict_timestamps


def gaitevents_to_framelist(phase_dict, lead_time):

    for key, value in phase_dict.items():
        phase_dict[key] = np.searchsorted(lead_time, value)

    l = np.zeros(len(lead_time))
    r = np.zeros(len(lead_time))
    l_len = get_phase_dict_len(phase_dict['l_heel_strike'], phase_dict['l_toe_off'])
    r_len = get_phase_dict_len(phase_dict['r_heel_strike'], phase_dict['r_toe_off'])
    toe_off_end = False
    heel_strike = None
    for i in range(l_len):
        try:
            heel_strike = phase_dict['l_heel_strike'][i]
        except IndexError:
            pass
        try:
            toe_off = phase_dict['l_toe_off'][i]
        except IndexError:
            toe_off = 99999999
            toe_off_end = True
        if heel_strike is None:
            print('No steps')
            exit()
        if heel_strike < toe_off:
            if not toe_off_end:
                l[heel_strike:toe_off] = True
            else:
                l[heel_strike:] = True
        elif heel_strike > toe_off:
            l[0:heel_strike] = False

    toe_off_end = False
    for i in range(r_len):
        try:
            heel_strike = phase_dict['r_heel_strike'][i]
        except IndexError:
            pass

        try:
            toe_off = phase_dict['r_toe_off'][i]
        except IndexError:
            toe_off = np.inf
            toe_off_end = True
        if heel_strike < toe_off:
            if not toe_off_end:
                r[heel_strike:toe_off] = True
            else:
                r[heel_strike:] = True
        elif heel_strike > toe_off:
            r[0:heel_strike] = False

    frames = np.empty(len(l), dtype=str)
    for index in range(len(l)):
        index = int(index)
        if l[index] == True and r[index] == True:
            frames[index] = 'd'
        elif l[index] == True and r[index] == False:
            frames[index] = 'l'
        elif r[index] == True and l[index] == False:
            frames[index] = 'r'

    return frames


def to_yaml(out, framelist=None, dictionary=None):
    if dictionary:
        for key, value in dictionary.items():
            dictionary[key] = value.flatten().tolist()
        f = open(out, 'w+')
        yaml.dump(dictionary, f, allow_unicode=True)
        f.close()
    elif all(framelist):
        f = open(out, 'w+')
        yaml.dump({'phases': framelist.tolist()}, f, allow_unicode=True)
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert from eurobench gait events file to a frame-based vector of gait events and vice versa')
    parser.add_argument('-g', '--gaitevents', type=argparse.FileType('r'), required=False, help='Provide a list of gait events [l_heel_strike, l_toe_off, r_heel_strike, r_toe_off] with corresponding timestamps.')
    parser.add_argument('-f', '--framelist', type=argparse.FileType('r'), required=False, help='Provide a list of all frames indicating the gait phase [l, r, d] each.')
    parser.add_argument('-t', '--leadtime', type=argparse.FileType('r'), required=True, help='Provide a .csv file containing a time column with the timeseries of the experiment.')
    parser.add_argument('-o', '--out', type=pathlib.Path, required=True, help='Provide the output path and filename.')
    args = parser.parse_args()
    if args.gaitevents and not args.framelist:
        print(f"Loaded '{args.gaitevents.name}'.")
        buffer = pd.read_csv(args.leadtime, sep=';')
        leadtime = buffer['time'].to_numpy().flatten()
        data = gaitevents_to_framelist(yaml.load(args.gaitevents, Loader=yaml.FullLoader), leadtime)
        to_yaml(args.out, framelist=data)
        print(f"Processed gaitevents into framelist at '{args.out}'. \nFinished.")
    elif args.framelist and not args.gaitevents:
        print(f"Loaded '{args.framelist.name}'.")
        buffer = pd.read_csv(args.leadtime, sep=';')
        leadtime = buffer['time'].to_numpy().flatten()
        data = framelist_to_gaitevents(yaml.load(args.framelist, Loader=yaml.FullLoader), leadtime)
        to_yaml(args.out, dictionary=data)
        print(f"Processed framelist into gaitevents at '{args.out}'. Finished.")
    else:
        args.print_help()
        sys.exit()
