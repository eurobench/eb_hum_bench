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

        if ftl and ftr:
            self.get_floor_contact_foot()
        # self.mechanical = self.mechanical()

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

    def get_floor_contact_foot(self):
        foot_l_floor_i = self.ftl.query('force_z > 30').index.values.tolist()
        foot_r_floor_i = self.ftr.query('force_z > 30').index.values.tolist()
        return foot_l_floor_i, foot_r_floor_i

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
        seg = exp.gait_segmentation()

        print('fin')


