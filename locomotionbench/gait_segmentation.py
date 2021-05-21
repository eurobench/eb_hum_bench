#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gait_segmentation.py:
Classify different phases of gait into single and double support phases.
"""
__author__ = ["Felix Aller", "Monika Harant"]
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = "Martin Felis"
__license__ = "BSD-2-Clause"
__version__ = "0.4"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

import numpy as np
import pandas as pd
from csaps import csaps
import rbdl
from locomotionbench.environment import FootContact
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
              (f.__name__, args, kw, te - ts))
        return result

    return wrap


def assign_phase(time, fl_pos_x_dot_cut, fl_pos_z_dot_cut, l_upper, fr_pos_x_dot_cut, fr_pos_z_dot_cut,
                 r_upper):
    phases = pd.DataFrame(columns=['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj'])
    fl_single = np.zeros(len(time), dtype=bool)
    fr_single = np.zeros(len(time), dtype=bool)
    double = np.zeros(len(time), dtype=bool)

    for index, value in time.itertuples():
        if fl_pos_x_dot_cut[index] != -1 and fl_pos_z_dot_cut[index] != -1 and l_upper[index] != -999:
            # and fl_vel_x_cut[j_] != -1
            fl_single[index] = True

        if fr_pos_x_dot_cut[index] != -1 and fr_pos_z_dot_cut[index] != -1 and r_upper[index] != -999:
            # and fr_vel_x_cut[j_] != -1:
            fr_single[index] = True

        if fl_single[index] and fr_single[index]:
            double[index] = True
            fl_single[index] = False
            fr_single[index] = False
        elif not fl_single[index] and not fr_single[index]:
            double[index] = True

    phases['fl_single'] = fl_single
    phases['fr_single'] = fr_single
    phases['double'] = double

    return phases


@timing
def gait_segmentation(robot_, experiment_):
    """
    segment the complete gait according to 3-phases: single support l/r and double support
    """
    pos = experiment_.files['pos']
    vel = experiment_.files['vel']
    acc = experiment_.files['acc']
    ftl = experiment_.files['ftl']
    ftr = experiment_.files['ftr']
    lead_time = experiment_.lead_time.to_numpy().flatten()

    l_upper = np.zeros(len(lead_time))
    r_upper = np.zeros(len(lead_time))
    smooth = 0.99

    # TODO: parameterize weight threshold

    up = -(robot_.mass * 0.2 * robot_.gravity[2])

    fl_ft = np.array(ftl['force_z'])
    fr_ft = np.array(ftr['force_z'])
    fl_ft_spl = csaps(lead_time, np.array(ftl['force_z']), smooth=smooth)
    fl_ft_smooth = fl_ft_spl(lead_time)
    fr_ft_spl = csaps(lead_time, np.array(ftr['force_z']), smooth=smooth)
    fr_ft_smooth = fr_ft_spl(lead_time)

    result = [get_pos_and_vel(robot_, np.ascontiguousarray(q), np.ascontiguousarray(qdot), np.ascontiguousarray(qddot))
              for q, qdot, qddot in zip(pos[experiment_.col_names].to_numpy(), vel[experiment_.col_names].to_numpy(),
                                        acc[experiment_.col_names].to_numpy())
              ]

    result = np.array(result)
    fl_pos = result[:, 0, :]
    fr_pos = result[:, 1, :]
    fl_vel = result[:, 2, :]
    fr_vel = result[:, 3, :]

    # Identify gait phase based on the force torque acting on the the feet
    l_upper = [i if j > up else -999 for i in fl_ft for j in fl_ft_smooth]
    r_upper = [i if j > up else -999 for i in fr_ft for j in fr_ft_smooth]

    # TODO: parameterize cut off parameters

    # Identify gait phase based on change of position of the feet in gait direction
    fl_pos_x = np.array([row[0] for row in fl_pos])
    fr_pos_x = np.array([row[0] for row in fr_pos])
    fl_pos_x_dot = np.gradient(fl_pos_x, np.array(lead_time))
    fr_pos_x_dot = np.gradient(fr_pos_x, np.array(lead_time))
    fl_pos_x_dot_cut = np.array([-1 if x >= 0.2 else x for x in fl_pos_x_dot])
    fr_pos_x_dot_cut = np.array([-1 if x >= 0.2 else x for x in fr_pos_x_dot])

    # Identify gait phase based on change of height of the feet
    fl_pos_z = np.array([row[2] for row in fl_pos])
    fr_pos_z = np.array([row[2] for row in fr_pos])
    fl_pos_z_dot = np.gradient(fl_pos_z, np.array(lead_time))
    fr_pos_z_dot = np.gradient(fr_pos_z, np.array(lead_time))
    fl_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fl_pos_z_dot])
    fr_pos_z_dot_cut = np.array([-1 if x >= 0.1 else x for x in fr_pos_z_dot])

    # Identify gait phase based on the velocity of feet
    # fl_vel_x = np.array([row[0] for row in fl_vel])
    # fr_vel_x = np.array([row[0] for row in fr_vel])
    # fl_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fl_vel_x])
    # fr_vel_x_cut = np.array([-1 if x >= 0.25 else x for x in fr_vel_x])
    # phases = pd.DataFrame(columns=['fl_single', 'fr_single', 'double', 'fl_obj', 'fr_obj'])

    phases = assign_phase(experiment_.lead_time, fl_pos_x_dot_cut, fl_pos_z_dot_cut,
                          np.array(l_upper), fr_pos_x_dot_cut, fr_pos_z_dot_cut, np.array(r_upper))
    return phases


def get_pos_and_vel(robot, q, qdot, qddot):
    rbdl.UpdateKinematics(robot.model, q, qdot, qddot)

    foot_contact_point_l = rbdl.CalcBodyToBaseCoordinates(robot.model, q,
                                                          robot.body_map.index(robot.l_foot) + 1,
                                                          np.array(robot.relative_sole_pos), False)
    foot_contact_point_r = rbdl.CalcBodyToBaseCoordinates(robot.model, q,
                                                          robot.body_map.index(robot.r_foot) + 1,
                                                          np.array(robot.relative_sole_pos), False)

    foot_velocity_l = rbdl.CalcPointVelocity(robot.model, q, qdot, robot.body_map.index(robot.l_foot) + 1,
                                             np.array(robot.relative_sole_pos), False)

    foot_velocity_r = rbdl.CalcPointVelocity(robot.model, q, qdot, robot.body_map.index(robot.r_foot) + 1,
                                             np.array(robot.relative_sole_pos), False)

    return [foot_contact_point_l, foot_contact_point_r, foot_velocity_l, foot_velocity_r]


def crop_start_end_phases(gait_phases_):
    #  find first occurrence of single support phase
    left = gait_phases_.fl_single.idxmax() - 1
    right = gait_phases_.fr_single.idxmax() - 1
    start_ds_end = min(left, right)
    #  find last occurrence of single support phase
    single = gait_phases_.fl_single
    left = len(single) - next(idx for idx, val in enumerate(single[::-1], 1) if val) + 1
    single = gait_phases_.fr_single
    right = len(single) - next(idx for idx, val in enumerate(single[::-1], 1) if val) + 1
    end_ds_start = max(left, right)
    return start_ds_end, end_ds_start


def crop_start_end_halfstep(gait_phases_, start_ds_end_, end_ds_start_):
    #  is left or right leg the first moving leg. find the last contact of this leg
    if gait_phases_.fl_single.loc[start_ds_end_ + 1]:
        r_f = gait_phases_.fl_single[:start_ds_end_]
    elif gait_phases_.fr_single.loc[start_ds_end_ + 1]:
        r_f = gait_phases_.fr_single[start_ds_end_ + 1:]
    else:
        return False
    # apply the same logic but go backwards
    if gait_phases_.fl_single.loc[end_ds_start_ - 1]:
        r_b = gait_phases_.fl_single.loc[end_ds_start_ - 1:0:-1]
    elif gait_phases_.fr_single.loc[end_ds_start_ - 1]:
        r_b = gait_phases_.fr_single.loc[end_ds_start_ - 1:0:-1]
    else:
        return False
    remove_front = r_f.idxmin() - 1
    remove_back = r_b.idxmin()
    return remove_front, remove_back


def create_contacts(gait_phases_, robot_, experiment_):
    left_single_support = gait_phases_.query('fl_single == True').index.tolist()
    right_single_support = gait_phases_.query('fr_single == True').index.tolist()
    double_support = gait_phases_.query('double == True').index.tolist()
    pos = experiment_.files['pos']
    vel = experiment_.files['vel']
    ftl = experiment_.files['ftl']
    ftr = experiment_.files['ftr']
    for index, value in experiment_.lead_time.itertuples():
        i_ = index
        q = np.array(pos.loc[i_].drop('time'))
        qdot = np.array(vel.loc[i_].drop('time'))
        if i_ in left_single_support:
            foot1 = FootContact(robot_.model, robot_.body_map.index(robot_.l_foot) + 1, robot_.relative_sole_pos,
                                robot_.sole_l,
                                robot_.foot_r_c, np.array(ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                                np.array(ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
            gait_phases_.loc[i_, ['fl_obj']] = foot1

        # are we looking at a single support phase (or double): prepare foot right contact
        elif i_ in right_single_support:

            foot1 = FootContact(robot_.model, robot_.body_map.index(robot_.r_foot) + 1, robot_.relative_sole_pos,
                                robot_.sole_r,
                                robot_.foot_r_c, np.array(ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                                np.array(ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
            gait_phases_.loc[i_, ['fr_obj']] = foot1


        # are we looking at a single support phase (or double): prepare foot contact of a support polygon for both feet
        elif i_ in double_support:
            foot1 = FootContact(robot_.model, robot_.body_map.index(robot_.l_foot) + 1, robot_.relative_sole_pos,
                                robot_.sole_l,
                                robot_.foot_r_c, np.array(ftl.loc[i_, ['force_x', 'force_y', 'force_z']]),
                                np.array(ftl.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
            foot2 = FootContact(robot_.model, robot_.body_map.index(robot_.r_foot) + 1, robot_.relative_sole_pos,
                                robot_.sole_r,
                                robot_.foot_r_c, np.array(ftr.loc[i_, ['force_x', 'force_y', 'force_z']]),
                                np.array(ftr.loc[i_, ['torque_x', 'torque_y', 'torque_z']]), q, qdot)
            foot1.set_cos(np.multiply(1 / 2, (foot1.cos + foot2.cos)))
            foot2.set_cos(np.multiply(1 / 2, (foot1.cos + foot2.cos)))
            gait_phases_.loc[i_, ['fl_obj']] = foot1
            gait_phases_.loc[i_, ['fr_obj']] = foot2

        return gait_phases_
