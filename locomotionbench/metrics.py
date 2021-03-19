#!/usr/bin/env python3
"""
@package locomotionbench
@file metrics.py
@author Felix Aller, Monika Harant
@brief compute several metrics based on gait segmentation for periodic walking motions
Copyright (C) 2020 Felix Aller
Distributed under the  BSD-2-Clause License.
"""
import pandas as pd
import numpy as np
import rbdl
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from shapely.geometry import MultiPoint
import matplotlib.pyplot as plt


def calc_cop(p_1, f_1, m_1, p_2=None, f_2=None, m_2=None):
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


def distance_point_vector(p1_, p2_, p3_):
    p1_ = np.array(p1_)
    p2_ = np.array(p2_)
    p3_ = np.array(p3_)
    return np.linalg.norm(np.cross(p2_ - p1_, p1_ - p3_)) / np.linalg.norm(p2_ - p1_)


class Foot:
    def __init__(self, model_=None, id_=None, rel_sole_pos_=None, corners_local_=None, foot_r_c_=None, force_=None,
                 moment_=None, q_=None, qdot_=None):

        self.id = id_
        self.corners_local = corners_local_
        self.foot_r_c = np.array(foot_r_c_)
        self.forces = force_
        self.moments = moment_
        if id_:
            self.cos, self.omega_f, self.corners_global = self.phase_dep_indicators(model_, q_, qdot_, rel_sole_pos_)
        else:
            self.cos, self.omega_f, self.corners_global = None, None, None

    def phase_dep_indicators(self, model__, q__, qdot__, rel_sole_pos__):
        corners_global = []
        cos = rbdl.CalcBodyToBaseCoordinates(model__, q__, self.id, np.array(rel_sole_pos__), True)
        omega_f = rbdl.CalcPointVelocity6D(model__, q__, qdot__, self.id, self.foot_r_c, False)[:3]
        for j_ in range(len(self.corners_local)):
            corners_global.append(
                rbdl.CalcBodyToBaseCoordinates(model__, q__, self.id, np.array(self.corners_local[j_]), True)[:2])
        corners_global = np.array(corners_global).flatten()
        return cos, omega_f, corners_global


class Metrics:
    def __init__(self, robot_, exp_):
        path = exp_['inputdir'] + '/' + exp_['project'] + '/' + exp_['run'] + '/' + exp_['trial'] + '/'
        model = robot_['modelpath'] + '/' + robot_['robotmodel']
        self.base_link = robot_['base_link']
        self.g = exp_['gravity']
        self.l_foot = robot_['foot_l']
        self.r_foot = robot_['foot_r']
        self.relative_sole_pos = robot_['sole_pos']
        self.leg_length = robot_['leg_length']
        self.mass = robot_['mass']
        self.sole_l = robot_['sole_shape_l']
        self.sole_r = robot_['sole_shape_r']
        #  TODO: really need this?
        self.foot_r_c = [0.061129, 0.003362, -0.004923]
        self.pos = pd.read_csv(path + exp_['files']['pos'], sep=exp_['separator'])
        self.vel = pd.read_csv(path + exp_['files']['pos'], sep=exp_['separator'])
        self.acc = pd.read_csv(path + exp_['files']['pos'], sep=exp_['separator'])
        self.trq = pd.read_csv(path + exp_['files']['pos'], sep=exp_['separator'])
        if exp_['files']['ftl'] and exp_['files']['ftr']:
            self.ftl = pd.read_csv(path + exp_['files']['ftl'], sep=exp_['separator'])
            self.ftr = pd.read_csv(path + exp_['files']['ftr'], sep=exp_['separator'])
        self.model = rbdl.loadModel(model, floating_base=True, verbose=False)
        self.dof = self.model.dof_count

        # File Header joint order should be equal to joint order from model file
        self.body_map = self.body_map_sorted()
        # TODO check order for all files in a smart way
        column_names = list(self.pos.columns)
        column_names.remove('time')
        if self.body_map != column_names:
            self.pos = self.pos.reindex(columns=['time'] + self.body_map)
            self.vel = self.vel.reindex(columns=['time'] + self.body_map)
            self.acc = self.acc.reindex(columns=['time'] + self.body_map)
            self.trq = self.trq.reindex(columns=['time'] + self.body_map)

        # TODO check if column time = lead time for all files
        self.lead_time = self.pos['time']
        self.foot_contacts = pd.DataFrame(columns=['fl_x1', 'fl_y1', 'fl_x2', 'fl_y2', 'fl_x3', 'fl_y3', 'fl_x4',
                                                   'fl_y4', 'fr_x1', 'fr_y1', 'fr_x2', 'fr_y2', 'fr_x3', 'fr_y3',
                                                   'fr_x4', 'fr_y4'])
        '''
        x1_____________x2
        |               |
        |     L_FOOT    |--->
        |               |
        x4_____________x3

        y1_____________y2
        |               |
        |    R_FOOT     |--->
        |               |
        y4_____________y3
        '''

        self.gait_segments = self.gait_segmentation()
        self.indicators = self.create_indicator_dataframe()

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

    def gait_segmentation(self):
        """
        segment the complete gait according to 3-phases: single support l/r and double support
        """
        fl_pos, fr_pos, fl_vel, fr_vel = [], [], [], []
        l_upper = np.zeros(len(self.lead_time))
        r_upper = np.zeros(len(self.lead_time))

        # TODO: parameterize weight threshold

        up = -(self.mass * 0.8 * self.g[2])

        fl_ft = np.array(self.ftl['force_z'])
        fr_ft = np.array(self.ftr['force_z'])

        for i_, value in self.lead_time.items():

            q = np.array(self.pos.loc[i_].drop('time'))
            qdot = np.array(self.vel.loc[i_].drop('time'))
            qddot = np.array(self.acc.loc[i_].drop('time'))

            rbdl.UpdateKinematics(self.model, q, qdot, qddot)

            foot_contact_point_l = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                  self.body_map.index(self.l_foot) + 1,
                                                                  np.array(self.relative_sole_pos), False)
            foot_contact_point_r = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                  self.body_map.index(self.r_foot) + 1,
                                                                  np.array(self.relative_sole_pos), False)

            foot_velocity_l = rbdl.CalcPointVelocity(self.model, q, qdot, self.body_map.index(self.l_foot) + 1,
                                                     np.array(self.relative_sole_pos), False)

            foot_velocity_r = rbdl.CalcPointVelocity(self.model, q, qdot, self.body_map.index(self.r_foot) + 1,
                                                     np.array(self.relative_sole_pos), False)
            fl_pos.append(foot_contact_point_l)
            fr_pos.append(foot_contact_point_r)
            fl_vel.append(foot_velocity_l)
            fr_vel.append(foot_velocity_r)

            # Identify gait phase based on the force torque acting on the the feet
            if fl_ft[i_] > up:
                l_upper[i_] = fl_ft[i_]
            else:
                l_upper[i_] = -1

            if fr_ft[i_] > up:
                r_upper[i_] = fr_ft[i_]
            else:
                r_upper[i_] = -1

        # TODO: parameterize cut off parameters

        # Identify gait phase based on change of position of the feet in gait direction
        fl_pos_x = np.array([row[0] for row in fl_pos])
        fr_pos_x = np.array([row[0] for row in fr_pos])
        fl_pos_x_dot = np.gradient(fl_pos_x, np.array(self.lead_time))
        fr_pos_x_dot = np.gradient(fr_pos_x, np.array(self.lead_time))
        fl_pos_x_dot_cut = np.array([-1 if x >= 0.1 else x for x in fl_pos_x_dot])
        fr_pos_x_dot_cut = np.array([-1 if x >= 0.1 else x for x in fr_pos_x_dot])

        # Identify gait phase based on change of height of the feet
        fl_pos_z = np.array([row[2] for row in fl_pos])
        fr_pos_z = np.array([row[2] for row in fr_pos])
        fl_pos_z_dot = np.gradient(fl_pos_z, np.array(self.lead_time))
        fr_pos_z_dot = np.gradient(fr_pos_z, np.array(self.lead_time))
        fl_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fl_pos_z_dot])
        fr_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fr_pos_z_dot])

        # Identify gait phase based on the velocity of feet
        fl_vel_x = np.array([row[0] for row in fl_vel])
        fr_vel_x = np.array([row[0] for row in fr_vel])
        fl_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fl_vel_x])
        fr_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fr_vel_x])

        segmentation = pd.DataFrame(columns=['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj'])
        fl_single = np.zeros(len(self.lead_time), dtype=bool)
        fr_single = np.zeros(len(self.lead_time), dtype=bool)
        double = np.zeros(len(self.lead_time), dtype=bool)

        for j_, value in self.lead_time.items():
            if fl_pos_x_dot_cut[j_] != -1 and fl_pos_z_dot_cut[j_] != -1 and fl_ft[
                j_] != -1:  # and fl_vel_x_cut[j_] != -1
                fl_single[j_] = True

            if fr_pos_x_dot_cut[j_] != -1 and fr_pos_z_dot_cut[j_] != -1 and fr_ft[
                j_] != -1:  # and fr_vel_x_cut[j_] != -1:
                fr_single[j_] = True

            if fl_single[j_] and fr_single[j_]:
                double[j_] = True
                fl_single[j_] = False
                fr_single[j_] = False

        # for k_ in range(len(self.lead_time)):
        #     if fl_single[k_] == True:
        #         lcolor_ = 'r'
        #     elif fl_single[k_] == 0:
        #         lcolor_ = 'b'
        #     else:
        #         lcolor_ = 'grey'
        #     if fr_single[k_] == True:
        #         rcolor_ = 'r'
        #     elif fr_single[k_] == 0:
        #         rcolor_ = 'b'
        #     else:
        #         rcolor_ = 'grey'
        #     if double[k_] == True:
        #         rcolor_ = 'g'
        #         lcolor_ = 'g'
        #
        #     plt.scatter(np.array(self.lead_time)[k_], fl_pos_x[k_], color=lcolor_, marker='x')
        #     plt.scatter(np.array(self.lead_time)[k_], fr_pos_x[k_], color=rcolor_, marker='x')
        # plt.show()

        segmentation['fl_single'] = fl_single
        segmentation['fr_single'] = fr_single
        segmentation['double'] = double

        return segmentation

    @staticmethod
    def create_indicator_dataframe(indicators=None):
        """
        :param indicators:
        :return:
        Accept a list of indicators and prepare an empty pandas dataframe to store all dof for all requested indidcators
        """

        """
        w_c:            normalized angular momentum
        h_c:            normalized angular momentum at com
        zmp:            zero moment point from dynamic model properties
        cop:            center of pressure from ft readings containing also double support phases
        cop_l:          cop of left foot
        cop_r:          cop of right foot

        dist_cop_bos_l: distance cop to boundary of support for left foot
        dist_cop_bos_r: distance cop to boundary of support for right foot

        csup:           center of support
        com_acc:        acceleration about com
        ang_vel:        angular velocity about com in n-direction (w0C0)
        ang_mom:        angular momentum about com
        v_gcom:         vector to ground projected com

        fpe:            foot placement estimator containing also double support phases
        fpe_l:          foot placement estimator for left leg
        fpe_r:          foot placement estimator for right leg
        dist_fpe_bos_l: distance fpe boundary of support for left foot
        dist_fpe_bos_r: distance fpe boundary of support for right foot

        cap:
        cap_l:
        cap_r:
        dist_cap_bos_l:
        dist_cap_bos_r:
        """
        concat = [['time']]
        if not indicators:
            indicators = ['com', 'cos', 'w_c', 'h_c', 'zmp', 'cop', 'fpe', 'cap', 'omega_f']
        if 'com' in indicators:
            columns = ['com_x', 'com_y', 'com_z', 'com_vel_x', 'com_vel_y', 'com_vel_z', 'com_acc_x', 'com_acc_y',
                       'com_acc_z']
            concat.append(columns)
        if 'cos' in indicators:  # center of support location
            columns = ['cos_x', 'cos_y', 'cos_z']
            concat.append(columns)
        if 'w_c' in indicators:  # normalized angular momentum
            columns = ['w_c_x', 'w_c_y', 'w_c_z']
            concat.append(columns)
        if 'h_c' in indicators:  # normalized angular momentum at com
            columns = ['h_c_x', 'h_c_y', 'h_c_z']
            concat.append(columns)
        if 'zmp' in indicators:
            columns = ['zmp_x', 'zmp_y', 'zmp_z', 'dist_zmp_bos']
            concat.append(columns)
        if 'cop' in indicators:
            columns = ['cop_x', 'cop_y', 'cop_z', 'dist_cop_bos']
            concat.append(columns)
        if 'fpe' in indicators:
            columns = ['fpe_x', 'fpe_y', 'fpe_z', 'dist_fpe_bos', 'fpe_err']
            concat.append(columns)
        if 'cap' in indicators:
            columns = ['cap_x', 'cap_y', 'cap_z', 'dist_cap_bos']
            concat.append(columns)
        if 'omega_f' in indicators:
            columns = ['omega_f_x', 'omega_f_y', 'omega_f_z']
            concat.append(columns)

        # TODO: Orbital E, Proj Err

        return pd.DataFrame(columns=[item for sublist in concat for item in sublist])

    def calc_metrics(self):
        """
        Calculate and return a set of indicators which were previously requested
        :return: self.indicators
        """

        self.indicators['time'] = self.lead_time
        zmp = np.zeros(3)

        balance_tk = rbdl.BalanceToolkit()
        omega_small = 1e-6
        fpe_output = rbdl.FootPlacementEstimatorInfo()

        left_single_support = self.gait_segments.query('fl_single == True').index.tolist()
        right_single_support = self.gait_segments.query('fr_single == True').index.tolist()
        double_support = self.gait_segments.query('double == True').index.tolist()

        for i_, value in self.lead_time.items():

            q = np.array(self.pos.loc[i_].drop('time'))
            qdot = np.array(self.vel.loc[i_].drop('time'))
            qddot = np.array(self.acc.loc[i_].drop('time'))

            rbdl.UpdateKinematics(self.model, q, qdot, qddot)

            # are we looking at a single support phase (or double): prepare foot left contact
            if i_ in left_single_support:
                foot1 = Foot(self.model, self.body_map.index(self.l_foot) + 1, self.relative_sole_pos, self.sole_l,
                             self.foot_r_c, np.array(self.ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                             np.array(self.ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
                cos = foot1.cos
                self.gait_segments.loc[i_, ['fl_obj']] = foot1
                foot2 = Foot()

            # are we looking at a single support phase (or double): prepare foot right contact
            elif i_ in right_single_support:
                foot1 = Foot(self.model, self.body_map.index(self.r_foot) + 1, self.relative_sole_pos, self.sole_r,
                             self.foot_r_c, np.array(self.ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                             np.array(self.ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
                cos = foot1.cos
                self.gait_segments.loc[i_, ['fr_obj']] = foot1
                foot2 = Foot()

            # are we looking at a single support phase (or double): prepare foot contact of a support polygon for both feet
            elif i_ in double_support:
                foot1 = Foot(self.model, self.body_map.index(self.l_foot) + 1, self.relative_sole_pos, self.sole_l,
                             self.foot_r_c, np.array(self.ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                             np.array(self.ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
                foot2 = Foot(self.model, self.body_map.index(self.r_foot) + 1, self.relative_sole_pos, self.sole_r,
                             self.foot_r_c, np.array(self.ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                             np.array(self.ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
                cos = np.multiply(1 / 2, (foot1.cos + foot2.cos))
                self.gait_segments.loc[i_, ['fl_obj']] = foot1
                self.gait_segments.loc[i_, ['fr_obj']] = foot2
            else:  # no contact detected
                print('error at: ', i_)
                continue

            # --- NORMALIZED ANGULAR MOMENTUM ABOUT COM --- #
            r_c, h_c, v_c, a_c, model_mass = self.calc_com(q, qdot, qddot)
            h_cn = h_c / (self.mass * self.leg_length ** 2)
            self.indicators.loc[i_, ['com_x', 'com_y', 'com_z']] = r_c
            self.indicators.loc[i_, ['com_vel_x', 'com_vel_y', 'com_vel_z']] = v_c
            self.indicators.loc[i_, ['com_acc_x', 'com_acc_y', 'com_acc_z']] = a_c
            self.indicators.loc[i_, ['h_c_x', 'h_c_y', 'h_c_z']] = h_cn
            # -------------------------------------------- #

            # --- KINETIC ENERGY --- #
            # self.ke.append(rbdl.CalcKineticEnergy(self.model, q, qdot))
            # ---------------------- #

            # --- FPE --- #
            balance_tk.CalculateFootPlacementEstimator(self.model, q, qdot, cos, np.array([0., 0., 1.]),
                                                       fpe_output, omega_small, False, False)
            fpe = fpe_output.r0F0
            self.indicators.loc[i_, ['fpe_x', 'fpe_y', 'fpe_z']] = fpe
            self.indicators.loc[i_, ['fpe_err']] = fpe_output.projectionError
            # ----------- #

            # --- CAP --- #
            # requires balance_tk from fpe
            cap = self.calc_cap(fpe_output.u, fpe_output.h, fpe_output.v0C0u, fpe_output.r0P0)
            self.indicators.loc[i_, ['cap_x', 'cap_y', 'cap_z']] = cap
            # ----------- #

            # --- COP --- #
            cop = calc_cop(foot1.cos, foot1.forces, foot1.moments, foot2.cos, foot2.forces, foot2.moments)
            self.indicators.loc[i_, ['cop_x', 'cop_y', 'cop_z']] = cop
            # ----------- #

            # --- ZMP --- #
            rbdl.CalcZeroMomentPoint(self.model, q, qdot, qddot, zmp, np.array([0., 0., 1.]), np.array([0., 0., 1.]),
                                     False)
            self.indicators.loc[i_, ['zmp_x', 'zmp_y', 'zmp_z']] = zmp
            # ----------- #

            # --- DISTANCES TO BOS --- #
            self.indicators.loc[i_, ['dist_zmp_bos']] = self.distance_to_support_polygon(q, zmp, foot1, foot2)
            self.indicators.loc[i_, ['dist_fpe_bos']] = self.distance_to_support_polygon(q, fpe, foot1, foot2)
            self.indicators.loc[i_, ['dist_cap_bos']] = self.distance_to_support_polygon(q, cap, foot1, foot2)
            # ------------------------ #

            # --- VARIOUS --- #
            self.indicators.loc[i_, ['cos_x', 'cos_y', 'cos_z']] = cos
            self.indicators.loc[
                i_, ['omega_f_x', 'omega_f_y', 'omega_f_z']] = foot1.omega_f  # in ds we just use the left leg
            # --------------- #

        return self.indicators

    def calc_com(self, q_, qdot_, qddot_):
        r_c_ = np.zeros(3)
        v_c_ = np.zeros(3)
        a_c_ = np.zeros(3)
        h_c_ = np.zeros(3)
        model_mass_ = rbdl.CalcCenterOfMass(self.model, q_, qdot_, r_c_, qddot_, v_c_, a_c_, h_c_, None, False)
        return r_c_, h_c_, v_c_, a_c_, model_mass_

    def distance_to_double_support(self, q_, poi_, foot1_, foot2_):
        '''
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
        '''

        foot1_ori_ = rbdl.CalcBodyWorldOrientation(self.model, q_, foot1_.id, True)

        point_ = rbdl.CalcBaseToBodyCoordinates(self.model, q_, foot1_.id, poi_)
        point_ = foot1_ori_.dot(point_)

        polygon = []

        for corner in foot1_.corners_local:
            cc = foot1_ori_.dot(corner)
            polygon.append((cc[0], cc[1]))

        for i in range(0, len(foot2_.corners_global), 2):
            cc = rbdl.CalcBaseToBodyCoordinates(self.model, q_, foot1_.id,
                                                np.array([foot2_.corners_global[i], foot2_.corners_global[i + 1], 0.]),
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

    def calc_cap(self, u_, h_, v_, gcom_):
        return gcom_ + np.multiply(v_ * math.sqrt(h_ / abs(self.g[2])), u_)
