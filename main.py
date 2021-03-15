import sys
import pandas as pd
import numpy as np
import rbdl
from sklearn.cluster import OPTICS
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
import hdbscan
import matplotlib.pyplot as plt
import math

PROJECT = "2021_02_19"
TOTAL_RUNS = 1
RUN = '24'
INTERVAL_SIZE = 20

MODEL = 'conf/reemc.lua'
BASE_LINK = ['base_link_tx', 'base_link_ty', 'base_link_tz', 'base_link_rz', 'base_link_ry', 'base_link_rx']

PARAMETERS = {'distance': 765
              }
MAX_TRQ = {'leg_left_1_joint': 42.7,
           'leg_left_2_joint': 64,
           'leg_left_3_joint': 55.7,
           'leg_left_4_joint': 138.3,
           'leg_left_5_joint': 80.9,
           'leg_left_6_joint': 64,
           'leg_right_1_joint': 42.7,
           'leg_right_2_joint': 64,
           'leg_right_3_joint': 55.7,
           'leg_right_4_joint': 138.3,
           'leg_right_5_joint': 80.9,
           'leg_right_6_joint': 64
           }

SOLE_POS = [0.12, 0., 0.]
L_FOOT = 'leg_left_6_joint'
R_FOOT = 'leg_right_6_joint'
GRAVITY = np.array([0., 0., -9.81])
LEG_LENGTH = 0.85853
MASS = 77.5
SOLE_L = [[0.12, 0.08, -0.09], [0.12, 0.08, 0.12], [0.12, -0.06, 0.12], [0.12, -0.06, -0.09]]
SOLE_R = [[0.12, 0.06, -0.09], [0.12, 0.06, 0.12], [0.12, -0.08, 0.12], [0.12, -0.08, -0.09]]


def calc_cop(p_1, f_1, m_1, p_2=None, f_2=None, m_2=None):
    cop = np.zeros(3)
    if p_2 and f_2 and m_2:
        cop[0] = (-m_1[1] - m_2[1] + (f_1[2] * p_1[0]) + (f_2[2] * p_2[0])) / (f_1[2] + f_2[2])
        cop[1] = (m_1[0] + m_2[0] + (f_1[2] * p_1[1]) + (f_2[2] * p_1[1])) / (f_1[2] + f_2[2])
        cop[2] = p_2[2]
    else:
        cop[0] = - (m_1[1] / f_1[2]) + p_1[0]
        cop[1] = p_1[1] + (m_1[0] / f_1[2])
        cop[2] = p_1[2]
    return cop


def calc_cap(self, u_, h_, v_, gcom_):
    return gcom_ + np.multiply(v_ * math.sqrt(h_ / abs(self.g[2])), u_)


def distance_point_vector(p1_, p2_, p3_):
    p1_ = np.array(p1_)
    p2_ = np.array(p2_)
    p3_ = np.array(p3_)
    return np.linalg.norm(np.cross(p2_ - p1_, p1_ - p3_)) / np.linalg.norm(p2_ - p1_)

class BENCH:
    def __init__(self, model, pos, vel, acc, trq, ftl=None, ftr=None, sep=';'):
        self.base_link = BASE_LINK
        self.pos = pd.read_csv(pos, sep=sep)
        self.vel = pd.read_csv(vel, sep=sep)
        self.acc = pd.read_csv(acc, sep=sep)
        self.trq = pd.read_csv(trq, sep=sep)
        if ftl and ftr:
            self.ftl = pd.read_csv(ftl, sep=sep)
            self.ftr = pd.read_csv(ftr, sep=sep)
        self.model = rbdl.loadModel(model, floating_base=True, verbose=False)
        self.dof = self.model.dof_count
        self.g = GRAVITY
        self.l_foot = L_FOOT
        self.r_foot = R_FOOT
        self.relative_sole_pos = SOLE_POS
        self.leg_length = LEG_LENGTH
        self.mass = MASS
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

        self.gait_segments = self.gait_segmentation()
        self.indicators = self.create_indicator_dataframe()
        # if ftl and ftr:
        #     self.get_floor_contact_foot()
        # # self.mechanical = self.mechanical()

    def body_map_sorted(self):
        orig = self.model.mBodyNameMap
        del orig['ROOT']
        del orig['base_link']
        resorted = {k: v for k, v in sorted(orig.items(), key=lambda item: item[1])}
        order = self.base_link
        for item in resorted:
            # TODO: replace link(model) with joint(file)
            order.append(item.strip().replace('_link', '_joint'))
        return order

    # def get_floor_contact_foot(self):
    #     foot_l_floor_i = self.ftl.query('force_z > 30').index.values.tolist()
    #     foot_r_floor_i = self.ftr.query('force_z > 30').index.values.tolist()
    #     return foot_l_floor_i, foot_r_floor_i

    def gait_segmentation(self):
        fl_pos, fr_pos, fl_vel, fr_vel = [], [], [], []
        l_upper = np.zeros(len(self.lead_time))
        r_upper = np.zeros(len(self.lead_time))

        # TODO: parameterize weight treshold

        up = -(self.mass * 0.8 * self.g)[2]

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

            if fl_ft[i_] > up:
                l_upper[i_] = fl_ft[i_]
            else:
                l_upper[i_] = -1

            if fr_ft[i_] > up:
                r_upper[i_] = fr_ft[i_]
            else:
                r_upper[i_] = -1

        # TODO: parameterize cut off parameters

        fl_pos_x = np.array([row[0] for row in fl_pos])
        fr_pos_x = np.array([row[0] for row in fr_pos])
        fl_pos_x_dot = np.gradient(fl_pos_x, np.array(self.lead_time))
        fr_pos_x_dot = np.gradient(fr_pos_x, np.array(self.lead_time))
        fl_pos_x_dot_cut = np.array([-1 if x >= 0.1 else x for x in fl_pos_x_dot])
        fr_pos_x_dot_cut = np.array([-1 if x >= 0.1 else x for x in fr_pos_x_dot])

        fl_pos_z = np.array([row[2] for row in fl_pos])
        fr_pos_z = np.array([row[2] for row in fr_pos])
        fl_pos_z_dot = np.gradient(fl_pos_z, np.array(self.lead_time))
        fr_pos_z_dot = np.gradient(fr_pos_z, np.array(self.lead_time))
        fl_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fl_pos_z_dot])
        fr_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fr_pos_z_dot])

        fl_vel_x = np.array([row[0] for row in fl_vel])
        fr_vel_x = np.array([row[0] for row in fr_vel])
        fl_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fl_vel_x])
        fr_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fr_vel_x])

        segmentation = pd.DataFrame(columns=['fl_single', 'fr_single', 'double'])
        fl_single = np.zeros(len(self.lead_time), dtype=bool)
        fr_single = np.zeros(len(self.lead_time), dtype=bool)
        double = np.zeros(len(self.lead_time), dtype=bool)

        for j_, value in self.lead_time.items():
            if fl_pos_x_dot_cut[j_] != -1 and fl_pos_z_dot_cut[j_] != -1 and fl_vel_x_cut[j_] != -1 and fl_ft[j_] != -1:
                fl_single[j_] = True

            if fr_pos_x_dot_cut[j_] != -1 and fr_pos_z_dot_cut[j_] != -1 and fr_vel_x_cut[j_] != -1 and fr_ft[j_] != -1:
                fr_single[j_] = True

            if fl_single[j_] == 1 and fr_single[j_] == 1:
                double[j_] = True

        # for k_ in range(len(self.lead_time)):
        #     if fl_single[k_] == 1:
        #         lcolor_ = 'r'
        #     elif fl_single[k_] == 0:
        #         lcolor_ = 'b'
        #     else:
        #         lcolor_ = 'grey'
        #     if fr_single[k_] == 1:
        #         rcolor_ = 'r'
        #     elif fr_single[k_] == 0:
        #         rcolor_ = 'b'
        #     else:
        #         rcolor_ = 'grey'
        #     if double[k_] == 1:
        #         rcolor_ = 'g'
        #         lcolor_ = 'g'
        #
        #     plt.scatter(timearray[k_], fl_pos_x[k_], color=lcolor_, marker='x')
        #     plt.scatter(timearray[k_], fr_pos_x[k_], color=rcolor_, marker='x')
        # plt.show()

        segmentation['fl_single'] = fl_single
        segmentation['fr_single'] = fr_single
        segmentation['double'] = double

        return segmentation

    def create_indicator_dataframe(self, indicators=None):
        '''
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
        '''
        concat = [['time']]
        if not indicators:
            indicators = ['com', 'cos', 'w_c', 'h_c', 'zmp', 'cop', 'fpe', 'cap', 'omega_f']
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
            columns = ['fpe_x', 'fpe_y', 'fpe_z', 'dist_fpe_bos', 'err']
            concat.append(columns)
        if 'cap' in indicators:
            columns = ['cap_x', 'cap_y', 'cap_z', 'dist_cap_bos']
            concat.append(columns)
        if 'omega_f' in indicators:
            columns = ['omega_f_x', 'omega_f_y', 'omega_f_z']
            concat.append(columns)

        # TODO: FOOT VEL, Orbital E, Proj Err, COM_VEL

        return pd.DataFrame(columns=[item for sublist in concat for item in sublist])

    def calc_indicators(self):

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
        #
        # fl_contacts, fr_contacts = self.get_floor_contact_foot()
        zmp = np.zeros(3)

        # ol, or, ur, ul
        foot_corners_l = SOLE_L
        foot_corners_r = SOLE_R
        balance_tk = rbdl.BalanceToolkit()
        fpe_output = rbdl.FootPlacementEstimatorInfo()

        left_single_support = self.gait_segments.query('fl_single == True').index.tolist()
        right_single_support = self.gait_segments.query('fr_single == True').index.tolist()
        double_support = self.gait_segments.query('double == True').index.tolist()

        left_single_support = [x for x in left_single_support if x not in double_support]
        right_single_support = [x for x in right_single_support if x not in double_support]

        for i, value in self.lead_time.items():
            q = np.array(self.pos.loc[i].drop('time'))
            qdot = np.array(self.vel.loc[i].drop('time'))
            qddot = np.array(self.acc.loc[i].drop('time'))
            center_of_support = np.array(3)
            rbdl.UpdateKinematics(self.model, q, qdot, qddot)
            corners = []
            single = False
            double = False
            # <-- Angular Momentum -->
            r_c, h_c, model_mass = self.calc_com(q, qdot)
            h_cn = h_c / (self.mass * self.leg_length ** 2)
            self.indicators['h_c_x', 'h_c_y', 'h_c_z'].loc[i] = h_cn
            # self.com.loc[i], model_mass = self.calc_com(q, qdot)
            # w_c = self.calc_normalized_angular_momentum(q, qdot, r_c, h_c)
            # self.w_c.loc[i] = w_c # normalized AM by Body Inertia
            # local AM at COM
            # <-- Angular Momentum

            # <-- Kinetic Energy -->
            # self.ke.append(rbdl.CalcKineticEnergy(self.model, q, qdot))
            # <-- Kinetic Energy

            # <-- ZMP -->
            # foot_contact_point_l, foot_contact_point_r = [False, False, False], [False, False, False]

            if i in left_single_support or double_support:
                single = True
                center_of_support_l = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                     self.body_map.index(self.l_foot) + 1,
                                                                     np.array(self.relative_sole_pos), True)
                center_of_support = center_of_support_l
                for j in range(len(foot_corners_l)):
                    corners.append(rbdl.CalcBodyToBaseCoordinates(self.model,
                                                                  q,
                                                                  self.body_map.index(self.l_foot) + 1,
                                                                  np.array(foot_corners_l[j]), True)[:2])
                corners = np.array(corners).flatten()
                self.foot_contacts.loc[
                    i, ['fl_x1', 'fl_y1', 'fl_x2', 'fl_y2', 'fl_x3', 'fl_y3', 'fl_x4', 'fl_y4']] = corners
            if i in right_single_support or double_support:
                single = True
                center_of_support_r = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                     self.body_map.index(self.r_foot) + 1,
                                                                     np.array(self.relative_sole_pos), True)
                center_of_support = center_of_support_r
                for j in range(len(foot_corners_r)):
                    corners.append(rbdl.CalcBodyToBaseCoordinates(self.model,
                                                                  q,
                                                                  self.body_map.index(self.r_foot) + 1,
                                                                  np.array(foot_corners_r[j]), True)[:2])
                corners = np.array(corners).flatten()
                self.foot_contacts.loc[
                    i, ['fr_x1', 'fr_y1', 'fr_x2', 'fr_y2', 'fr_x3', 'fr_y3', 'fr_x4', 'fr_y4']] = corners
            if i in double_support:
                double = True
                center_of_support = np.multiply(1 / 2, (center_of_support_l + center_of_support_r))
            if not center_of_support:
                print('error at: ', i)
                continue

            if double:
                cop = calc_cop(p_1=center_of_support_l,
                               f_1=np.array(self.ftl.loc[i, ['force_x', 'force_y', 'force_z']]),
                               m_1=np.array(self.ftl.loc[i, ['torque_x', 'torque_y', 'torque_z']]),
                               p_2=center_of_support_r,
                               f_2=np.array(self.ftr.loc[i, ['force_x', 'force_y', 'force_z']]),
                               m_2=np.array(self.ftr.loc[i, ['torque_x', 'torque_y', 'torque_z']]))
            else:
                cop = calc_cop(p_1=center_of_support, f_1=self.ftl.loc[i, ['force_x', 'force_y', 'force_z']],
                               m_1=self.ftl.loc[i, ['torque_x', 'torque_y', 'torque_z']])
            self.indicators['cop_x', 'cop_y', 'cop_z'].loc[i] = cop
            self.indicators['omega_f_x', 'omega_f_y', 'omega_f_z'].loc[i] = rbdl.CalcPointVelocity6D(self.model,
                                                                                                     q,
                                                                                                     qdot,
                                                                                                     self.body_map.index(
                                                                                                         self.l_foot) + 1,
                                                                                                     # np.array(self.relative_sole_pos), True
                                                                                                     np.array(
                                                                                                         [0.061129,
                                                                                                          0.003362,
                                                                                                          -0.004923]),
                                                                                                     False
                                                                                                     )[:3]

            balance_tk.CalculateFootPlacementEstimator(self.model, q, qdot, center_of_support, np.array([0., 0., 1.]),
                                                       fpe_output, False, False)
            self.indicators['fpe_x', 'fpe_y', 'fpe_z'].loc[i] = fpe_output.r0F0
            self.indicators['err'].loc[i] = fpe_output.projectionError
            self.indicators['cap_l_x', 'cap_l_y', 'cap_l_z'].loc[i] = calc_cap(fpe_output.u, fpe_output.h,
                                                                               fpe_output.v0C0u, fpe_output.r0P0)

            rbdl.CalcZeroMomentPoint(self.model, q, qdot, qddot, zmp, np.array([0., 0., 1.]),np.array([0., 0., 1.]),
                                     False)
            self.indicators['zmp_x', 'zmp_y', 'zmp_z'].loc[i] = zmp

        #     self.dist_zmp_fpe.loc[i] = np.linalg.norm(fpe[:2] - zmp[:2])
        #     self.dist_zmp_kinematic_fpe.loc[i] = np.linalg.norm(fpe[:2] - zmp_kinetic[:2])
        #     self.dist_fpe_cp.loc[i] = np.linalg.norm(fpe[:2] - cp[:2])
        #     self.dist_zmp_gcom.loc[i] = np.linalg.norm(zmp[:2] - fpe_input.r0P0[:2])
        #     self.dist_zmp_kinematic_gcom.loc[i] = np.linalg.norm(zmp_kinetic[:2] - fpe_input.r0P0[:2])
        #     self.dist_zmp_kinematic_zmp.loc[i] = np.linalg.norm(zmp_kinetic[:2] - zmp[:2])
        #
        #     invalid_borders, case = self.check_point_within_support_polygon(q, zmp, fl_contact, fr_contact,
        #                                                                     foot_corners_l, foot_corners_r)
        #     self.dist_zmp_bos.loc[i] = self.distance2edge(zmp, corners_l, corners_r, case, invalid_borders)
        #     invalid_borders, case = self.check_point_within_support_polygon(q, zmp_kinetic, fl_contact,
        #                                                                     fr_contact, foot_corners_l,
        #                                                                     foot_corners_r)
        #     self.dist_zmp_kinematic_bos.loc[i] = self.distance2edge(zmp_kinetic, corners_l, corners_r, case,
                                                                    invalid_borders)
        #     invalid_borders, case = self.check_point_within_support_polygon(q, fpe, fl_contact,
        #                                                                     fr_contact, foot_corners_l, foot_corners_r)
        #     self.dist_fpe_bos.loc[i] = self.distance2edge(fpe, corners_l, corners_r, case, invalid_borders)
        #     invalid_borders, case = self.check_point_within_support_polygon(q, cp, fl_contact,
        #                                                                     fr_contact, foot_corners_l, foot_corners_r)
        #     self.dist_cp_bos.loc[i] = self.distance2edge(cp, corners_l, corners_r, case, invalid_borders)
        #
        #     self.dist_zmp_base_l.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                  self.body_map.index(self.l_foot) + 1, zmp_l)
        #     self.dist_zmp_base_r.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                  self.body_map.index(self.r_foot) + 1, zmp_r)
        #     self.dist_fpe_base_l.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                  self.body_map.index(self.l_foot) + 1, fpe_l)
        #     self.dist_fpe_base_r.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                  self.body_map.index(self.r_foot) + 1, fpe_r)
        #     self.dist_cp_base_l.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                 self.body_map.index(self.l_foot) + 1, cp_l)
        #     self.dist_cp_base_r.loc[i] = rbdl.CalcBaseToBodyCoordinates(self.model, q,
        #                                                                 self.body_map.index(self.r_foot) + 1, cp_r)
        #     self.cos.loc[i, ['cos_x', 'cos_y', 'cos_z']] = center_of_support
        #
        #     self.orbital_energy.loc[i] = self.orbital_enrgy(fpe_input.r0C0, zmp, fpe_input.v0C0)
        #     # <-- ZMP
        # print('Boundaries')
        # print(get_intervals(self.foot_contacts['fl_x1'].to_numpy(), False))
        # print(get_intervals(self.foot_contacts['fr_x1'].to_numpy(), False))
        # # boundaries_r = get_intervals(experiment_.foot_contacts['fr_x1'].to_numpy(), True)
        # print('----')

    def calc_com(self, q_, qdot_):
        r_c_ = np.zeros(3)
        h_c_ = np.zeros(3)
        model_mass_ = rbdl.CalcCenterOfMass(self.model, q_, qdot_, r_c_, angular_momentum=h_c_, update_kinematics=False)
        return r_c_, h_c_, model_mass_

    def check_point_within_support_polygon(self, q, point, f1_contact, f2_contact, corners_l, corners_r):
        invalid_borders = []
        case = -1
        if f1_contact and f2_contact:
            xr_1 = rbdl.CalcBodyToBaseCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1,
                                                  np.array(corners_r[1]))
            xr_1_l = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.l_foot) + 1, xr_1)

            case = 0 # left foot behind right foot

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
                case = 1    # right foot behind left foot
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
            case = 2 # right foot is contact foot
            point_r = rbdl.CalcBaseToBodyCoordinates(self.model, q, self.body_map.index(self.r_foot) + 1, point)
            if corners_r[2][1] <= point_r[1] and point_r[1] <= corners_r[0][1] and corners_r[0][2] <= point_r[2] and point_r[2] <= corners_r[2][2]:
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

    def distance2edge(self, point, contacts_l_, contacts_r_, case = -1, invalid_borders=[]):
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

        if case == 2:   # right foot is contact foot
            for i in range(len(indices) - 1):
                dist_edge.append(distance_point_vector(corner_contacts_r[indices[i]], corner_contacts_r[indices[i + 1]], point))
        elif case == 3: # left foot is contact foot
            for i in range(len(indices) - 1):
                dist_edge.append(distance_point_vector(corner_contacts_l[indices[i]], corner_contacts_l[indices[i + 1]], point))

        elif case == 0: # left foot behind right foot
            dist_edge.append(distance_point_vector(corner_contacts_l[0], corner_contacts_l[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[1], corner_contacts_r[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[1], corner_contacts_r[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[2], corner_contacts_r[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[3], corner_contacts_l[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[3], corner_contacts_l[0], point))
        elif case == 1: # right foot behind left foot
            dist_edge.append(distance_point_vector(corner_contacts_l[0], corner_contacts_l[1], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[1], corner_contacts_l[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_l[2], corner_contacts_r[2], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[2], corner_contacts_r[3], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[3], corner_contacts_r[0], point))
            dist_edge.append(distance_point_vector(corner_contacts_r[0], corner_contacts_l[0], point))
        else:
            print("case does not exist: minimum edge", case)

        for i in invalid_borders:
            dist_edge[i] = -dist_edge[i]

        if (case == 2 or case == 3) and np.amin(dist_edge) > 0.07:
            print(dist_edge)

        # if np.amin(dist_edge) < -0.1:
        #      print("smaller", dist_edge, case)

        return np.amin(dist_edge)

if __name__ == '__main__':
    total_runs = TOTAL_RUNS
    experiment = {}
    experiment_norm = {}
    for i in range(total_runs):
        i += 1
        if i > total_runs:
            continue
        POS = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'pos.csv'
        VEL = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'vel.csv'
        ACC = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'acc.csv'
        TRQ = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'trq.csv'
        FTL = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'ftl.csv'
        FTR = 'input/' + PROJECT + '/' + RUN + '/' + str(i) + '/' + 'ftr.csv'

        exp = BENCH(MODEL, POS, VEL, ACC, TRQ, FTL, FTR)
        exp.calc_indicators()

        print('fin')
