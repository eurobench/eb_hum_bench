import pandas as pd
import numpy as np
import rbdl
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt


PROJECT = "2021_02_19"
TOTAL_RUNS = 1
RUN = '16'
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

class Kinetics:
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
        self.start_time = 0
        self.end_time = 59.95985794067383
        self.total_dist = 7.65
        self.g = np.array([0., 0., -9.81])
        self.l_foot = 'leg_left_6_joint'
        self.r_foot = 'leg_right_6_joint'
        self.relative_sole_pos = [0.12, 0., 0.]
        #self.com_height = 0.81
        self.leg_length = 0.85853
        self.mass = 77.5
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
            # replace link(model) with joint(file)
            order.append(item.strip().replace('_link', '_joint'))
        return order

    def get_floor_contact_foot(self):
        foot_l_floor_i = self.ftl.query('force_z > 30').index.values.tolist()
        foot_r_floor_i = self.ftr.query('force_z > 30').index.values.tolist()
        return foot_l_floor_i, foot_r_floor_i

    def cluster(self):
        self.in_support_poly = pd.DataFrame(columns=['support'])

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

        self.foot_contacts = pd.DataFrame(
            columns=['fl_x1', 'fl_y1', 'fl_x2', 'fl_y2', 'fl_x3', 'fl_y3', 'fl_x4', 'fl_y4', 'fr_x1', 'fr_y1', 'fr_x2',
                     'fr_y2', 'fr_x3', 'fr_y3', 'fr_x4', 'fr_y4'])
        self.ke = []
        self.center_of_support = []

        fl_contacts, fr_contacts = self.get_floor_contact_foot()
        zmp_kinetic = np.zeros(3)
        fl_point, fr_point = [], []

        for i, value in self.lead_time.items():
            q = np.array(self.pos.loc[i].drop('time'))
            qdot = np.array(self.vel.loc[i].drop('time'))
            qddot = np.array(self.acc.loc[i].drop('time'))

            rbdl.UpdateKinematics(self.model, q, qdot, qddot)
            foot_contact_point_l = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                  self.body_map.index(self.l_foot) + 1,
                                                                  np.array(self.relative_sole_pos), True)
            foot_contact_point_r = rbdl.CalcBodyToBaseCoordinates(self.model, q,
                                                                  self.body_map.index(self.r_foot) + 1,
                                                                  np.array(self.relative_sole_pos), True)
            fl_point.append(foot_contact_point_l)
            fr_point.append(foot_contact_point_r)

        fl_y = np.array([row[2] for row in fl_point])
        fr_y = [row[2] for row in fr_point]

        rng = np.random.RandomState(41)
        X = rng.randn(1, 100)

        timestate = np.arange(0, len(self.lead_time))
        timearray = np.array(self.lead_time)
        time = len(np.array(self.lead_time))
        #data = np.column_stack((timearray, fl_y))

        gradient = np.gradient(fl_y, timearray)

        distances = np.array([Coords2 - Coords1 for Coords1, Coords2 in zip(timearray[:-1], timearray[1:])])

        avg = np.average(distances)
        min = np.min(distances)
        max = np.max(distances)
        print(max)

        ft = np.array(self.ftl['force_z'])
        # data = np.column_stack((timearray, gradient))
        data = np.column_stack((timearray, ft))


        # clusterer = OPTICS(min_samples=2, min_cluster_size=10, max_eps=0.021, eps = 0.021)
        clusterer = OPTICS(min_samples=5, min_cluster_size=30, max_eps=5, eps = 5)
        clusterer.fit(data)
        #clusterer = DBSCAN(eps=0.05, min_samples=20).fit(dataset)
        #cluster_labels = clusterer.fit_predict(data)


        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'pink', 'skyblue', 'yellow', 'peru', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'skyblue', 'yellow', 'peru', 'r', 'g', 'b', 'c', 'm', 'y', 'pink', 'skyblue', 'yellow', 'peru']

        for i in range(len(data)):
            if clusterer.labels_[i] == -1:
                color_ = 'grey'
            else:
                color_ = colors[clusterer.labels_[i]]
            plt.scatter(timearray[i], ft[i], color=color_)
        plt.show()

        plt.plot(self.lead_time, fl_y, 'x')
        plt.show()
        plt.plot(self.lead_time, fl_y, '-', self.lead_time, fr_y, '-')
        plt.show()
        plt.plot(self.lead_time, np.array(self.pos['base_link_tz']))
        plt.show()


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

        exp = Kinetics(MODEL, POS, VEL, ACC, TRQ, FTL, FTR)
        exp.cluster()
