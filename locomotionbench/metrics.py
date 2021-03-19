#!/usr/bin/env python3
"""
@package locomotionbench
@file metrics.py
@author Felix Aller
@brief compute several metrics based on gait segmentation for periodic walking motions
Copyright (C) 2020 Felix Aller
Distributed under the  BSD-2-Clause License.
"""
import pandas as pd
import numpy as np
import sophus as sp
import rbdl
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# PROJECT = "2021_02_19"
# TOTAL_RUNS = 1
# RUN = '16'
#
#
# MODEL = 'conf/reemc.lua'
# BASE_LINK = ['base_link_tx', 'base_link_ty', 'base_link_tz', 'base_link_rz', 'base_link_ry', 'base_link_rx']
#
# PARAMETERS = {'distance': 765
#               }
# MAX_TRQ = {'leg_left_1_joint': 42.7,
#            'leg_left_2_joint': 64,
#            'leg_left_3_joint': 55.7,
#            'leg_left_4_joint': 138.3,
#            'leg_left_5_joint': 80.9,
#            'leg_left_6_joint': 64,
#            'leg_right_1_joint': 42.7,
#            'leg_right_2_joint': 64,
#            'leg_right_3_joint': 55.7,
#            'leg_right_4_joint': 138.3,
#            'leg_right_5_joint': 80.9,
#            'leg_right_6_joint': 64
#            }
#
# SOLE_POS = [0.12, 0., 0.]
# L_FOOT = 'leg_left_6_joint'
# R_FOOT = 'leg_right_6_joint'
# GRAVITY = np.array([0., 0., -9.81])
# LEG_LENGTH = 0.85853
# MASS = 77.5
# SOLE_L = [[0.12, 0.08, -0.09], [0.12, 0.08, 0.12], [0.12, -0.06, 0.12], [0.12, -0.06, -0.09]]
# SOLE_R = [[0.12, 0.06, -0.09], [0.12, 0.06, 0.12], [0.12, -0.08, 0.12], [0.12, -0.08, -0.09]]


def calc_cop(p_1, f_1, m_1, p_2=[], f_2=[], m_2=[]):
    cop = np.zeros(3)
    # single support
    if all(v is None for v in p_2) and all(v is None for v in f_2) and all(v is None for v in m_2):
        cop[0] = - (m_1[1] / f_1[2]) + p_1[0]
        cop[1] = p_1[1] + (m_1[0] / f_1[2])
        cop[2] = p_1[2]
    # double support
    else:
        cop[0] = (-m_1[1] - m_2[1] + (f_1[2] * p_1[0]) + (f_2[2] * p_2[0])) / (f_1[2] + f_2[2])
        cop[1] = (m_1[0] + m_2[0] + (f_1[2] * p_1[1]) + (f_2[2] * p_1[1])) / (f_1[2] + f_2[2])
        cop[2] = p_2[2]
    return cop

def calc_grf(f_1, m_1, f_2=[], m_2=[]):
    grf = np.zeros(4)
    if all(v is None for v in f_2) and all(v is None for v in m_2):
        grf[0] = abs(f_1[0]) / f_1[2]
        grf[1] = abs(f_1[1]) / f_1[2]
        grf[2] = abs(m_1[0]) / f_1[2]
        grf[3] = abs(m_1[1]) / f_1[2]
    else:
        grf[0] = (abs(f_1[0]) / f_1[2]) + (abs(f_2[0]) / f_2[2])
        grf[1] = (abs(f_1[1]) / f_1[2]) + (abs(f_2[1]) / f_2[2])
        grf[2] = (abs(m_1[0]) / f_1[2]) + (abs(m_2[0]) / f_2[2])
        grf[3] = (abs(m_1[1]) / f_1[2]) + (abs(m_2[1]) / f_2[2])
    return grf

def distance_point_vector(p1_, p2_, p3_):
    p1_ = np.array(p1_)
    p2_ = np.array(p2_)
    p3_ = np.array(p3_)
    return np.linalg.norm(np.cross(p2_ - p1_, p1_ - p3_)) / np.linalg.norm(p2_ - p1_)


class Metrics:
    def __init__(self, robot_, exp_):
        path = exp_['inputdir'] + '/' + exp_['project'] + '/' + exp_['run'] + '/' + exp_['trial'] + '/'
        model = robot_['modelpath'] + '/' + robot_['robotmodel']
        self.base_link = robot_['base_link']
        self.g = exp_['gravity']
        self.l_foot = robot_['foot_l']
        self.r_foot = robot_['foot_r']
        self.torso_link = robot_['torso_link']
        self.relative_sole_pos = robot_['sole_pos']
        self.leg_length = robot_['leg_length']
        self.mass = robot_['mass']
        self.sole_l = robot_['sole_shape_l']
        self.sole_r = robot_['sole_shape_r']

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

        segmentation = pd.DataFrame(columns=['fl_single', 'fr_single', 'double'])
        fl_single = np.zeros(len(self.lead_time), dtype=bool)
        fr_single = np.zeros(len(self.lead_time), dtype=bool)
        double = np.zeros(len(self.lead_time), dtype=bool)

        for j_, value in self.lead_time.items():
            if fl_pos_x_dot_cut[j_] != -1 and fl_pos_z_dot_cut[j_] != -1  and fl_ft[j_] != -1:# and fl_vel_x_cut[j_] != -1
                fl_single[j_] = True

            if fr_pos_x_dot_cut[j_] != -1 and fr_pos_z_dot_cut[j_] != -1 and fr_ft[j_] != -1:# and fr_vel_x_cut[j_] != -1:
                fr_single[j_] = True

            if fl_single[j_] and fr_single[j_]:
                double[j_] = True

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
            indicators = ['com', 'cos', 'w_c', 'h_c', 'zmp', 'cop', 'fpe', 'cap', 'base', 'distance', 'impact', 'fc_vel', 'grf']
        if 'com' in indicators:
            columns = ['com_x', 'com_y', 'com_z', 'com_acc_x', 'com_acc_y', 'com_acc_z']
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
        if 'base' in indicators:
            columns = ['base_orientation_error_x', 'base_orientation_error_y', 'base_orientation_error_z']
            concat.append(columns)
        if 'distance' in indicators:
            columns = ['distance_covered', 'n_steps', 'normalized_dist_steps']
            concat.append(columns)
        if 'impact' in indicators:
            columns = ['impact']
            concat.append(columns)
        if 'fc_vel' in indicators:
            columns = ['fc_vel_ang_x', 'fc_vel_ang_y', 'fc_vel_ang_z', 'fc_vel_lin_x', 'fc_vel_lin_y', 'fc_vel_lin_z']
            concat.append(columns)
        if 'grf' in indicators:
            columns = ['f_mu_x', 'f_mu_y', 'f_dx', 'f_dy']
            concat.append(columns)

        # TODO: FOOT VEL, Orbital E, Proj Err, COM_VEL

        return pd.DataFrame(columns=[item for sublist in concat for item in sublist])

    def calc_metrics(self):
        """
        Calculate and return a set of indicators which were previously requested
        :return: self.indicators
        """
        #
        # fl_contacts, fr_contacts = self.get_floor_contact_foot()
        zmp = np.zeros(3)

        # ol, or, ur, ul
        foot_corners_l = self.sole_l
        foot_corners_r = self.sole_r
        balance_tk = rbdl.BalanceToolkit()
        omegaSmall = 1e-6
        fpe_output = rbdl.FootPlacementEstimatorInfo()

        left_single_support = self.gait_segments.query('fl_single == True').index.tolist()
        right_single_support = self.gait_segments.query('fr_single == True').index.tolist()
        double_support = self.gait_segments.query('double == True').index.tolist()

        left_single_support = [x for x in left_single_support if x not in double_support]
        right_single_support = [x for x in right_single_support if x not in double_support]

        distance_traveled, n_steps, normalized_dist_steps = self.n_steps_normalized_by_leg_distance(left_single_support, right_single_support, double_support)

        for i_, value in self.lead_time.items():

            q = np.array(self.pos.loc[i_].drop('time'))
            qdot = np.array(self.vel.loc[i_].drop('time'))
            qddot = np.array(self.acc.loc[i_].drop('time'))
            center_of_support = np.array(3)
            rbdl.UpdateKinematics(self.model, q, qdot, qddot)
            single_l, single_r, double = False, False, False
            # <-- Angular Momentum -->
            r_c, h_c, model_mass = self.calc_com(q, qdot)
            h_cn = h_c / (self.mass * self.leg_length ** 2)
            self.indicators.loc[i_, ['h_c_x', 'h_c_y', 'h_c_z']] = h_cn
            # self.com.loc[i], model_mass = self.calc_com(q, qdot)
            # w_c = self.calc_normalized_angular_momentum(q, qdot, r_c, h_c)
            # self.w_c.loc[i] = w_c # normalized AM by Body Inertia
            # local AM at COM
            # <-- Angular Momentum

            # <-- Kinetic Energy -->
            # self.ke.append(rbdl.CalcKineticEnergy(self.model, q, qdot))
            # <-- Kinetic Energy

            center_of_support_l, center_of_support_r = [False, False, False], [False, False, False]
            corners_l = []
            corners_r = []
            omega_f = []

            # are we looking at a single support phase (or double): prepare foot left contact
            if i_ in left_single_support:
                single_l = True
                center_of_support_l = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                     self.body_map.index(self.l_foot) + 1,
                                                                     np.array(self.relative_sole_pos), True)

                omega_f = rbdl.CalcPointVelocity6D(self.model, q, qdot, self.body_map.index(self.l_foot) + 1, np.array(
                    [0.061129,
                     0.003362,
                     -0.004923]), False)
                center_of_support = center_of_support_l


                for j_ in range(len(foot_corners_l)):
                    corners_l.append(rbdl.CalcBodyToBaseCoordinates(self.model,
                                                                  q,
                                                                  self.body_map.index(self.l_foot) + 1,
                                                                  np.array(foot_corners_l[j_]), True)[:2])
                corners_l = np.array(corners_l).flatten()
                self.foot_contacts.loc[
                    i_, ['fl_x1', 'fl_y1', 'fl_x2', 'fl_y2', 'fl_x3', 'fl_y3', 'fl_x4', 'fl_y4']] = corners_l

            # are we looking at a single support phase (or double): prepare foot right contact
            elif i_ in right_single_support:
                single_r = True
                center_of_support_r = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                     self.body_map.index(self.r_foot) + 1,
                                                                     np.array(self.relative_sole_pos), True)
                omega_f = rbdl.CalcPointVelocity6D(self.model, q, qdot, self.body_map.index(self.r_foot) + 1, np.array(
                    [0.061129,
                     0.003362,
                     -0.004923]), False)
                center_of_support = center_of_support_r

                for j_ in range(len(foot_corners_r)):
                    corners_r.append(rbdl.CalcBodyToBaseCoordinates(self.model,
                                                                  q,
                                                                  self.body_map.index(self.r_foot) + 1,
                                                                  np.array(foot_corners_r[j_]), True)[:2])
                corners_r = np.array(corners_r).flatten()
                self.foot_contacts.loc[
                    i_, ['fr_x1', 'fr_y1', 'fr_x2', 'fr_y2', 'fr_x3', 'fr_y3', 'fr_x4', 'fr_y4']] = corners_r

            # are we looking at a single support phase (or double): prepare foot contact of a support polygon for both feet
            elif i_ in double_support:
                double = True
                center_of_support = np.multiply(1 / 2, (center_of_support_l + center_of_support_r))

                omega_lf = rbdl.CalcPointVelocity6D(self.model, q, qdot, self.body_map.index(self.l_foot) + 1, np.array(
                    [0.061129,
                     0.003362,
                     -0.004923]), False)
                omega_rf = rbdl.CalcPointVelocity6D(self.model, q, qdot, self.body_map.index(self.r_foot) + 1, np.array(
                    [0.061129,
                     0.003362,
                     -0.004923]), False)
                omega_f = omega_lf + omega_rf

            if all(v is None for v in center_of_support):
                print('error at: ', i_)
                continue

            if double:
                cop = calc_cop(p_1=center_of_support_l,
                               f_1=np.array(self.ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                               m_1=np.array(self.ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]),
                               p_2=center_of_support_r,
                               f_2=np.array(self.ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                               m_2=np.array(self.ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]))
                grf = calc_grf(f_1=np.array(self.ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                               m_1=np.array(self.ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]),
                               f_2=np.array(self.ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                               m_2=np.array(self.ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]))
            else:
                grf = calc_grf(f_1=self.ftl.loc[i_, ['force_x', 'force_y', 'force_z']], m_1=self.ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']])

            # calculate all the indicators
            self.indicators.loc[i_, ['cop_x', 'cop_y', 'cop_z']] = cop
            self.indicators.loc[i_, ['f_mu_x', 'f_mu_y', 'f_dx', 'f_dy']] = grf
            self.indicators.loc[i_, ['fc_vel_ang_x', 'fc_vel_ang_y', 'fc_vel_ang_z', 'fc_vel_lin_x', 'fc_vel_lin_y', 'fc_vel_lin_z']] = omega_f

            balance_tk.CalculateFootPlacementEstimator(self.model, q, qdot, center_of_support_l, np.array([0., 0., 1.]),
                                                       fpe_output, omegaSmall, False, False)
            fpe = fpe_output.r0F0

            self.indicators.loc[i_, ['fpe_x', 'fpe_y', 'fpe_z']] = fpe
            self.indicators.loc[i_, ['fpe_err']] = fpe_output.projectionError
            cap = self.calc_cap(fpe_output.u, fpe_output.h, fpe_output.v0C0u, fpe_output.r0P0)
            self.indicators.loc[i_, ['cap_x', 'cap_y', 'cap_z']] = cap

            rbdl.CalcZeroMomentPoint(self.model, q, qdot, qddot, zmp, np.array([0., 0., 1.]), np.array([0., 0., 1.]),
                                     False)
            self.indicators.loc[i_, ['zmp_x', 'zmp_y', 'zmp_z']] = zmp

            # calculate distances from indicators to bos
            invalid_borders, case = self.check_point_within_support_polygon(q, zmp, single_l,
                                                                            single_r, foot_corners_l,
                                                                            foot_corners_r)
            self.indicators.loc[i_, ['dist_zmp_bos']] = self.distance2edge(zmp, corners_l, corners_r,
                                                                           case, invalid_borders)

            invalid_borders, case = self.check_point_within_support_polygon(q, fpe, single_l,
                                                                            single_r, foot_corners_l,
                                                                            foot_corners_r)
            self.indicators.loc[i_, ['dist_fpe_bos']] = self.distance2edge(fpe, corners_l, corners_r, case,
                                                                           invalid_borders)

            invalid_borders, case = self.check_point_within_support_polygon(q, cap, single_l,
                                                                            single_r, foot_corners_l,
                                                                            foot_corners_r)
            self.indicators.loc[i_, ['dist_cap_bos']] = self.distance2edge(cap, corners_l, corners_r, case,
                                                                           invalid_borders)

            self.indicators.loc[i_, ['base_orientation_error_x', 'base_orientation_error_y', 'base_orientation_error_z']] = self.calc_base_orientation_error(q)

        return self.indicators

    def calc_com(self, q_, qdot_):
        r_c_ = np.zeros(3)
        h_c_ = np.zeros(3)
        #  model, q, qdot, com, qddot, com_velocity, com_acceleration, angular_momentum
        model_mass_ = rbdl.CalcCenterOfMass(self.model, q_, qdot_, r_c_, angular_momentum=h_c_,
                                            update_kinematics=False)
        return r_c_, h_c_, model_mass_

    def check_point_within_support_polygon(self, q, point, fl_contact, fr_contact, corners_l, corners_r):
        invalid_borders = []
        case = -1
        if fl_contact and fr_contact:
            xr_1 = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1,
                                                  np.array(corners_r[1]))
            xr_1_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, xr_1)

            case = 0  # left foot behind right foot

            if xr_1_l[2] >= 0:

                xr_3 = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1,
                                                      np.array(corners_r[3]))
                xr_3_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, xr_3)
                point_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, point)

                t_1 = (point_l[1] - corners_l[1][1]) / (xr_1_l[1] - corners_l[1][1])
                y_1 = corners_l[1][2] + t_1 * (xr_1_l[2] - corners_l[1][2])

                t_3 = (point_l[1] - corners_l[3][1]) / (xr_3_l[1] - corners_l[3][1])
                y_3 = corners_l[3][2] + t_3 * (xr_3_l[2] - corners_l[3][2])

                if point_l[1] <= corners_l[0][1] and point_l[2] <= y_1 and point_l[2] <= xr_1_l[2] \
                        and point_l[1] >= xr_3_l[1] and point_l[2] >= y_3 and point_l[2] >= corners_l[0][2]:
                    return invalid_borders, case
                else:
                    if point_l[1] > corners_l[0][1]:
                        invalid_borders.append(0)
                    if point_l[2] > y_1:
                        invalid_borders.append(1)
                    if point_l[2] > xr_1_l[2]:
                        invalid_borders.append(2)
                    if point_l[1] < xr_3_l[1]:
                        invalid_borders.append(3)
                    if point_l[2] < y_3:
                        invalid_borders.append(4)
                    if point_l[2] < corners_l[0][2]:
                        invalid_borders.append(5)
            else:
                case = 1  # right foot behind left foot
                xr_0 = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1,
                                                      np.array(corners_r[0]))
                xr_0_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, xr_0)

                xr_2 = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1,
                                                      np.array(corners_r[2]))
                xr_2_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, xr_2)
                point_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, point)

                t_0 = (point_l[1] - corners_l[2][1]) / (xr_2_l[1] - corners_l[2][1])
                y_0 = corners_l[2][2] + t_0 * (xr_2_l[2] - corners_l[2][2])

                t_2 = (point_l[1] - corners_l[0][1]) / (xr_0_l[1] - corners_l[0][1])
                y_2 = corners_l[0][2] + t_2 * (xr_0_l[2] - corners_l[0][2])

                if point_l[1] <= corners_l[0][1] and point_l[2] <= corners_l[1][2] and point_l[2] <= y_0 \
                        and point_l[1] >= xr_2_l[1] and point_l[2] >= xr_0_l[2] and point_l[2] >= y_2:
                    return invalid_borders, case
                else:
                    if point_l[1] > corners_l[0][1]:
                        invalid_borders.append(0)
                    if point_l[2] > corners_l[1][2]:
                        invalid_borders.append(1)
                    if point_l[2] > y_0:
                        invalid_borders.append(2)
                    if point_l[1] < xr_2_l[1]:
                        invalid_borders.append(3)
                    if point_l[2] < xr_0_l[2]:
                        invalid_borders.append(4)
                    if point_l[2] < y_2:
                        invalid_borders.append(5)

        else:
            case = 2  # single support
            point_r = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1, point)
            if corners_r[2][1] <= point_r[1] <= corners_r[0][1] and corners_r[0][2] <= point_r[2] <= corners_r[2][2]:
                return invalid_borders, case
            else:
                if corners_r[2][1] > point_r[1]:
                    invalid_borders.append(2)
                if point_r[1] > corners_r[0][1]:
                    invalid_borders.append(0)
                if corners_r[0][2] > point_r[2]:
                    invalid_borders.append(3)
                if point_r[2] > corners_r[2][2]:
                    invalid_borders.append(1)

        return invalid_borders, case

    @staticmethod
    def distance2edge(point, contacts_l_, contacts_r_, case=-1, invalid_borders=None):
        if invalid_borders is None:
            invalid_borders = []
        dist_edge = []
        indices = [0, 1, 2, 3, 0]
        point = point[:2]
        corner_contacts_l = np.array([[contacts_l_[0], contacts_l_[1]],
                                      [contacts_l_[2], contacts_l_[3]],
                                      [contacts_l_[4], contacts_l_[5]],
                                      [contacts_l_[6], contacts_l_[7]],
                                      ])
        corner_contacts_r = np.array([[contacts_r_[0], contacts_r_[1]],
                                      [contacts_r_[2], contacts_r_[3]],
                                      [contacts_r_[4], contacts_r_[5]],
                                      [contacts_r_[6], contacts_r_[7]],
                                      ])

        if case == 2:  # single support
            for i_ in range(len(indices) - 1):
                dist_edge.append(
                    distance_point_vector(corner_contacts_r[indices[i_]], corner_contacts_r[indices[i_ + 1]], point))

        elif case == 0:  # left foot behind right foot
            dist_edge.append(distance_point_vector(corner_contacts_l[0], corner_contacts_l[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[1], corner_contacts_r[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[1], corner_contacts_r[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[2], corner_contacts_r[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[3], corner_contacts_l[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[3], corner_contacts_l[0], point))
        elif case == 1:  # right foot behind left foot
            dist_edge.append(distance_point_vector(corner_contacts_l[0], corner_contacts_l[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[1], corner_contacts_l[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[2], corner_contacts_r[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[2], corner_contacts_r[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[3], corner_contacts_r[0], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[0], corner_contacts_l[0], point))
        else:
            print("case does not exist: minimum edge", case)

        for i_ in invalid_borders:
            dist_edge[i_] = -dist_edge[i_]

        if case == 2 and np.amin(dist_edge) > 0.07:
            print(dist_edge)

        return np.amin(dist_edge)

    def calc_cap(self, u_, h_, v_, gcom_):
        return gcom_ + np.multiply(v_ * math.sqrt(h_ / abs(self.g[2])), u_)

    def calc_base_orientation_error(self, q):
        base_orient = rbdl.CalcBodyWorldOrientation(self.model, q, self.body_map.index(self.torso_link) + 1).transpose()
        r_base_orient = R.from_matrix(base_orient)
        r_identity = R.from_quat([0, 0, 0, 1])
        r_error = r_identity * r_base_orient.inv()
        s_error = sp.SO3(r_error.as_matrix())
        error = sp.SO3.log(s_error)
        return error

    def n_steps_normalized_by_leg_distance(self, left_single_support, right_single_support, double_support):
        n_steps = 0
        support_type = 'DS'
        prev_p = np.array([0., 0., 0.])
        distance_traveled = 0.0
        
        for i_, value in self.lead_time.items():
            #print(support_type)
            q = np.array(self.pos.loc[i_].drop('time'))
            if i_ in double_support:
                support_type = 'DS'
            elif i_ in left_single_support:
                if support_type is not 'LS':
                    p = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, np.array([0., 0., 0.]))
                    distance_traveled += p[0] - prev_p[0]
                    prev_p = p
                    n_steps += 1
                support_type = 'LS'
            elif i_ in right_single_support:
                if support_type is not 'RS':
                    p = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1, np.array([0., 0., 0.]))
                    distance_traveled += p[0] - prev_p[0]
                    prev_p = p
                    n_steps += 1
                support_type = 'RS'

        return distance_traveled, n_steps, distance_traveled / (n_steps * self.leg_length)

