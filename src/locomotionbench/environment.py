#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment.py:
Creates the experimental environment by parsing the data output files, robot information and robot-environment contacts
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig", "Martin Felis"]
__license__ = "BSD-2-Clause"
__version__ = "0.4"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

import sys

import rbdl
import yaml
import pandas as pd
import numpy as np
import os
import argparse
import pathlib
from csaps import csaps
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPoint
from src.locomotionbench.performance_indicator import timing
from src.script.gait_event_conversion import gaitevents_to_framelist


def parse_args(args):
    parser = argparse.ArgumentParser(description='Process EUROBENCH File Format')
    parser.add_argument('--conf', type=argparse.FileType('r'), required=True)
    parser.add_argument('--model', type=argparse.FileType('r'), required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    parser.add_argument('--gait', type=argparse.FileType('r'), required=True)
    parser.add_argument('--pos', help='Position File containing 6D-Freeflyer and Joint Angles', type=argparse.FileType('r'), required=True)
    parser.add_argument('--vel', type=argparse.FileType('r'))
    parser.add_argument('--acc', type=argparse.FileType('r'))
    parser.add_argument('--trq', type=argparse.FileType('r'))
    parser.add_argument('--ftl', type=argparse.FileType('r'))
    parser.add_argument('--ftr', type=argparse.FileType('r'))
    return parser.parse_args(args)


# Robot class to collect all the information for the used robot to expose it to performance calculation routine
class Robot:
    def __init__(self, args):
        # open yaml file containing the required information
        conf = yaml.load(args.conf, Loader=yaml.FullLoader)
        # robot model, used by most of the pi
        self.model = rbdl.loadModel(args.model.name, floating_base=True, verbose=False)
        self.gait_events = yaml.load(args.gait, Loader=yaml.FullLoader)
        self.base_link = conf['base_link']  # base link or root of the kinematic chain
        self.gravity = conf['gravity']  # gravity as a list of x, y, z
        self.l_foot = conf['foot_l']  # link name of the left foot in model
        self.r_foot = conf['foot_r']  # link name of the right foot in model
        self.torso_link = conf['torso_link']  # link name of the torso in model
        self.relative_sole_pos = conf['sole_pos']  # relative position of the robot sole with respect to the foot link
        self.leg_length = conf['leg_length']  # leg length of the robot for normalization
        self.mass = conf['mass']  # mass of the robot for normalization
        self.sole_l = conf['sole_shape_l']  # shape of the left sole of the robot: distances between corner points
        self.sole_r = conf['sole_shape_r']  # shape of the left sole of the robot: distances between corner points
        #  TODO: really need this?
        self.foot_r_c = [0.061129, 0.003362, -0.004923]  # position of the com of the foot
        self.body_map = self.body_map_sorted()  # sort body map according to body ids
        self.phases = None  # init gait phases
        self.cos = pd.DataFrame(columns=['x', 'y', 'z'], dtype='float')  # init data frame for center of support
        self.step_list = {'fl_single': None,
                          'fr_single': None,
                          'fl_double': None,
                          'fr_double': None}

    def body_map_sorted(self):
        orig = self.model.mBodyNameMap
        del orig['ROOT']
        del orig['base_link']
        resorted = {k: v for k, v in sorted(orig.items(), key=lambda item__: item__[1])}
        order = self.base_link
        for item_ in resorted:
            # TODO: replace link(model) with joint(file)
            order.append(item_.strip().replace('_link', '_joint'))
        return order

    def get_body_map(self):
        return self.body_map

        # if applicable and requested: cut of double support phases at start and end and additionally also the first and
        # last halfstep
    def truncate_gait(self, remove_ds=False, remove_hs=False):
        truncate = [None, None]
        if remove_ds is True:
            start_ds_end, end_ds_start = self.crop_start_end_phases()
            truncate = [start_ds_end, end_ds_start]
            if remove_hs is True:
                start_ds_hs_end, end_ds_hs_start = self.crop_start_end_halfstep(start_ds_end, end_ds_start)
                truncate = [start_ds_hs_end, end_ds_hs_start]

        # create index lists of individual steps
        for key in self.step_list:
            self.step_list[key] = self.create_step_list(self.phases.query(f'{key} == True').index.tolist(), truncate)

        return truncate

    @staticmethod
    def create_step_list(phases, crop=None):
        if not phases:
            return None
        step_size = 1
        step_list = np.split(phases, np.where(np.diff(phases) != step_size)[0] + 1)
        return step_list

    def get_pos_and_vel(self, q, qdot):
        # calculate foot position of left and right foot
        foot_contact_point_l = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                              self.body_map.index(self.l_foot) + 1,
                                                              np.array(self.relative_sole_pos), True)
        foot_contact_point_r = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                              self.body_map.index(self.r_foot) + 1,
                                                              np.array(self.relative_sole_pos), True)
        # calculate foot velocity of left and right foot
        foot_velocity_l = rbdl.CalcPointVelocity(self.model, q, qdot, self.body_map.index(self.l_foot) + 1,
                                                 np.array(self.relative_sole_pos), True)

        foot_velocity_r = rbdl.CalcPointVelocity(self.model, q, qdot, self.body_map.index(self.r_foot) + 1,
                                                 np.array(self.relative_sole_pos), True)

        return [foot_contact_point_l, foot_contact_point_r, foot_velocity_l, foot_velocity_r]

    def crop_start_end_phases(self):
        #  find first occurrence of single support phase
        left = self.phases.fl_single.idxmax() - 1
        right = self.phases.fr_single.idxmax() - 1
        start_ds_end = min(left, right)
        #  find last occurrence of single support phase
        single = self.phases.fl_single
        left = len(single) - next(idx for idx, val in enumerate(single[::-1], 1) if val) + 1
        single = self.phases.fr_single
        right = len(single) - next(idx for idx, val in enumerate(single[::-1], 1) if val) + 1
        end_ds_start = max(left, right)
        return start_ds_end, end_ds_start

    def crop_start_end_halfstep(self, start_ds_end_, end_ds_start_):
        #  is left or right leg the first moving leg. find the last contact of this leg
        if self.phases.fl_single.loc[start_ds_end_ + 1]:
            r_f = self.phases.fl_single[:start_ds_end_]
        elif self.phases.fr_single.loc[start_ds_end_ + 1]:
            r_f = self.phases.fr_single[start_ds_end_ + 1:]
        else:
            return False
        # apply the same logic but go backwards
        if self.phases.fl_single.loc[end_ds_start_ - 1]:
            r_b = self.phases.fl_single.loc[end_ds_start_ - 1:0:-1]
        elif self.phases.fr_single.loc[end_ds_start_ - 1]:
            r_b = self.phases.fr_single.loc[end_ds_start_ - 1:0:-1]
        else:
            return False
        remove_front = r_f.idxmin() - 1
        remove_back = r_b.idxmin()
        return remove_front, remove_back

    def truncate_phases(self, crop):
        self.phases = self.phases.truncate(before=crop[0]+1, after=crop[1])

    def truncate_step_list(self, crop):
        for key in self.step_list:
            temp = self.step_list[key]
            buffer = []
            for item in range(len(temp)):
                list = temp[item]
                if any(x > crop[1] for x in list) or any(x < crop[0] for x in list):
                    continue
                else:
                    temp2 = ([value - crop[0]-1 for value in temp[item]])
                    buffer.append(temp2)
            self.step_list[key] = buffer

    @timing
    def create_contacts(self, experiment_):
        left_single_support = self.phases.query('fl_single == True').index.tolist()
        right_single_support = self.phases.query('fr_single == True').index.tolist()
        double_support = self.phases.query('double == True').index.tolist()
        pos = experiment_.files['pos']
        # vel = experiment_.files['vel']
        no_force_torque = False
        if experiment_.files_not_provided.count('ftl') == 0 and experiment_.files_not_provided.count('ftr') == 0:
            ftl = experiment_.files['ftl']
            ftr = experiment_.files['ftr']
        else:
            no_force_torque = True
        for index, value in experiment_.lead_time.itertuples():
            cos = None
            q = np.array(pos.loc[index].drop('time'))
            # qdot = np.array(vel.loc[index].drop('time'))

            if not no_force_torque:
                force_l = np.array(ftl.loc[index, ['force_x', 'force_y', 'force_z']])
                torque_l = np.array(ftl.loc[index, ['torque_x', 'torque_y', 'torque_z']])
                force_r = np.array(ftr.loc[index, ['force_x', 'force_y', 'force_z']])
                torque_r = np.array(ftr.loc[index, ['torque_x', 'torque_y', 'torque_z']])
            else:
                force_l, torque_l, force_r, torque_r = None, None, None, None
            if index in left_single_support:
                foot1 = FootContact(self.body_map.index(self.l_foot) + 1, self.relative_sole_pos,
                                    self.sole_l,
                                    self.foot_r_c, force_l, torque_l
                                    )
                foot2 = FootContact()
                self.phases.loc[index, ['fl_obj']] = foot1
                self.phases.loc[index, ['fr_obj']] = foot2
                cos = foot1.get_cos(self.model, q)
            # are we looking at a single support phase (or double): prepare foot right contact
            elif index in right_single_support:

                foot1 = FootContact(self.body_map.index(self.r_foot) + 1, self.relative_sole_pos,
                                    self.sole_r,
                                    self.foot_r_c, force_r, torque_r)
                foot2 = FootContact()
                self.phases.loc[index, ['fr_obj']] = foot1
                self.phases.loc[index, ['fl_obj']] = foot2
                cos = foot1.get_cos(self.model, q)
            # are we looking at a single support phase (or double): prepare foot contact of a support polygon for both feet
            elif index in double_support:
                foot1 = FootContact(self.body_map.index(self.l_foot) + 1, self.relative_sole_pos,
                                    self.sole_l,
                                    self.foot_r_c, force_l, force_r)
                foot2 = FootContact(self.body_map.index(self.r_foot) + 1, self.relative_sole_pos,
                                    self.sole_r,
                                    self.foot_r_c, force_l, force_r)
                cos = np.multiply(1 / 2, (foot1.get_cos(self.model, q) + foot2.get_cos(self.model, q)))
                self.phases.loc[index, ['fl_obj']] = foot1
                self.phases.loc[index, ['fr_obj']] = foot2
            self.cos.loc[index, ['x', 'y', 'z']] = cos

    # @staticmethod
    # def assign_phase(time, fl_pos_x_dot_cut, fl_pos_z_dot_cut, l_upper, fr_pos_x_dot_cut, fr_pos_z_dot_cut,
    #                  r_upper):
    def assign_phase(self, leadtime):
        phases = pd.DataFrame(
            columns=['fl_single', 'fr_single', 'double', 'fl_double', 'fr_double', 'fl_obj', 'fr_obj'])
        frames = gaitevents_to_framelist(self.gait_events, leadtime)
        fl_single = np.zeros(len(frames), dtype=bool)
        fr_single = np.zeros(len(frames), dtype=bool)
        double = np.zeros(len(frames), dtype=bool)
        prev_l = np.zeros(len(frames), dtype=bool)
        prev_r = np.zeros(len(frames), dtype=bool)
        prev = None
        for i in range(len(frames)):
            if frames[i] == 'd':
                double[i] = True
                if prev:
                    if prev == 'l':
                        prev_l[i] = True
                    if prev == 'r':
                        prev_r[i] = True
            elif frames[i] == 'l':
                fl_single[i] = True
                prev = frames[i]
            elif frames[i] == 'r':
                fr_single[i] = True
                prev = frames[i]

        phases['fl_single'] = fl_single
        phases['fr_single'] = fr_single
        phases['double'] = double
        phases['fl_double'] = prev_l
        phases['fr_double'] = prev_r
        self.phases = phases

    def distance_to_double_support(self, q_, poi_, foot1_, foot2_):
        """
        needs module shapely
                    3-------2           2-------3
                    |       |           |       |
                    | foot2 |           | foot2 |
        2-------3   4-------1      or   1-------4  3-------2
        |       |                                  |       |
        | foot1 |                                  | foot1 |
        1-------4                                  4-------1
        :param q_: joint positions and global position and orientation
        :param poi_: point of interest to be checked whether inside of support polygon
        :param foot1_: containing all information about foot1 (body id, corners in body and global coordinates
        :param foot2_: containing all information about foot2 (body id, corners in body and global coordinates
        :return:
        """

        foot1_ori_ = rbdl.CalcBodyWorldOrientation(self.model, q_, foot1_.id, True)

        point_ = rbdl.CalcBaseToBodyCoordinates(self.model, q_, foot1_.id, poi_)
        point_ = foot1_ori_.dot(point_)

        polygon = []

        # corners_global_1 = foot1_.get_global_corners(self.model, q_)
        corners_global_2 = foot2_.get_global_corners(self.model, q_)

        for corner in foot1_.corners_local:
            cc = foot1_ori_.dot(corner)
            polygon.append((cc[0], cc[1]))

        for i in range(0, len(corners_global_2), 2):
            cc = rbdl.CalcBaseToBodyCoordinates(self.model, q_, foot1_.id,
                                                np.array([corners_global_2[i], corners_global_2[i + 1], 0.]),
                                                False)
            cc = foot1_ori_.dot(cc)
            polygon.append((cc[0], cc[1]))

        point_shapely = Point(point_[0], point_[1])
        multipoint = MultiPoint(polygon)
        polygon_shapely = multipoint.convex_hull

        distance = point_shapely.distance(polygon_shapely.boundary)

        if polygon_shapely.contains(point_shapely):
            return distance
        else:
            return -distance

    def distance_to_support_polygon(self, q_, poi_, foot1_, foot2_=None):

        if foot2_.id is None:
            foot1_ori_ = rbdl.CalcBodyWorldOrientation(self.model, q_, foot1_.id, True)
            # point of interest with respect to the frame of foot1
            point_ = rbdl.CalcBaseToBodyCoordinates(self.model, q_, foot1_.id, poi_)
            # transforming to the orientation of the base frame
            point_ = foot1_ori_.dot(point_)

            polygon = []

            for corner in foot1_.corners_local:
                cc = foot1_ori_.dot(corner)
                polygon.append((cc[0], cc[1]))

            point_shapely = Point(point_[0], point_[1])
            polygon_shapely = Polygon(polygon)

            # shortest distance to boundary of polygon
            distance = point_shapely.distance(polygon_shapely.boundary)

            if polygon_shapely.contains(point_shapely):
                return distance
            else:
                return -distance
        else:
            return self.distance_to_double_support(q_, poi_, foot1_, foot2_)


class FootContact:
    def __init__(self, id_=None, rel_sole_pos_=None, corners_local_=None, foot_r_c_=None, force_=None,
                 moment_=None):
        self.id = id_
        self.corners_local = corners_local_
        self.foot_r_c = np.array(foot_r_c_)
        self.forces = force_
        self.moment = moment_
        self.rel_sole_pos = rel_sole_pos_
        # self.moments = moment_
        # if id_:
        #     self.cos, self.omega_v, self.corners_global = self.phase_dep_indicators(model_, q_, qdot_, rel_sole_pos_)
        # else:
        #     self.cos, self.omega_v, self.corners_global = None, None, None

    def set_cos(self, cos):
        self.cos = cos

    def get_cos(self, model_, q_, ):
        self.cos = rbdl.CalcBodyToBaseCoordinates(model_, q_, self.id, np.array(self.rel_sole_pos), True)
        return self.cos

    def get_omega_v(self, model_, q_, qdot_):
        return rbdl.CalcPointVelocity6D(model_, q_, qdot_, self.id, self.foot_r_c, True)

    def get_global_corners(self, model_, q_):
        corners_global = []
        for j_ in range(len(self.corners_local)):
            corners_global.append(
                rbdl.CalcBodyToBaseCoordinates(model_, q_, self.id, np.array(self.corners_local[j_]), True)[:2])
        return np.array(corners_global).flatten()

    # def phase_dep_indicators(self, model_, q_, qdot_, rel_sole_pos_):
    #     corners_global = []
    #     cos = rbdl.CalcBodyToBaseCoordinates(model_, q_, self.id, np.array(rel_sole_pos_), True)
    #     omega_v = rbdl.CalcPointVelocity6D(model_, q_, qdot_, self.id, self.foot_r_c, False)
    #     for j_ in range(len(self.corners_local)):
    #         corners_global.append(
    #             rbdl.CalcBodyToBaseCoordinates(model_, q_, self.id, np.array(self.corners_local[j_]), True)[:2])
    #     corners_global = np.array(corners_global).flatten()
    #     return cos, omega_v, corners_global


class Experiment:
    def __init__(self, args, separator=';'):

        self.files = {}
        self.files_not_provided = []
        self.files_not_provided.append(self.open_file('pos', args.pos, separator))
        self.files_not_provided.append(self.open_file('vel', args.vel, separator))
        self.files_not_provided.append(self.open_file('acc', args.acc, separator))
        self.files_not_provided.append(self.open_file('trq', args.trq, separator))
        self.files_not_provided.append(self.open_file('ftl', args.ftl, separator))
        self.files_not_provided.append(self.open_file('ftr', args.ftr, separator))

        # self.files['conditions'] = yaml.load(file_names_[6], Loader=yaml.FullLoader)
        # TODO check order for all files in a smart way
        self.col_names = list(self.get_file('pos').columns)
        self.col_names.remove('time')

        # TODO check if column time = lead time for all files
        self.lead_time = self.get_file('pos').loc[:, ['time']]

    def are_columns_matching(self, body_map):
        if body_map != self.col_names:
            return False
        return True

    def reindex_columns(self, body_map):
        # TODO check order for all files in a smart way
        self.files['pos'] = self.get_file('pos').reindex(columns=['time'] + body_map)
        if 'vel' not in self.files_not_provided:
            self.files['vel'] = self.get_file('vel').reindex(columns=['time'] + body_map)
        if 'acc' not in self.files_not_provided:
            self.files['acc'] = self.get_file('acc').reindex(columns=['time'] + body_map)
        if 'trq' not in self.files_not_provided:
            self.files['trq'] = self.get_file('trq').reindex(columns=['time'] + body_map)

    def open_file(self, file_type, file, separator):
        if file is None:
            return file_type
        try:
            self.files[file_type] = pd.read_csv(file, sep=separator)
        except FileNotFoundError as err:
            print(f"{Color.WARNING}WARNING: {err.strerror} '{err.filename}' for {file_type} - is that correct?)")
            return file_type

    def get_file(self, file_name):
        try:
            file = self.files[file_name]
        except KeyError:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            Color.err_print(
                f"{file_name} file not found but required.--- {fname}, line: {exc_tb.tb_lineno}{Color.ENDC}")
            sys.exit()
        return file

    def truncate_lead_time(self, crop):
        self.lead_time = self.lead_time.truncate(before=crop[0]+1, after=crop[1])

    def truncate_data(self, crop):
        for key in self.files:
            self.files[key] = self.files[key].truncate(before=crop[0]+1, after=crop[1])

class Color:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def green_print(text):
        print(f"{Color.OKGREEN}{text}{Color.ENDC}")

    @staticmethod
    def cyan_print(text):
        print(f"{Color.OKCYAN}{text}{Color.ENDC}")

    @staticmethod
    def err_print(text):
        print(f"{Color.FAIL}{text}{Color.ENDC}")

    @staticmethod
    def warning_print(text):
        print(f"{Color.WARNING}{text}{Color.ENDC}")
