#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
i3sa.py:
Launches the corresponding performance indicators for experimental data retrieved from the EUROBENCH I3SA Protocol.
"""
__author__ = "Felix Aller"
__copyright__ = "Copyright 2021, EUROBENCH Project"
__credits__ = ["Monika Harant", "Adrià Roig"]
__license__ = "BSD-2"
__version__ = "0.4"
__maintainer__ = "Felix Aller"
__email__ = "felix.aller@ziti.uni-heidelberg.de"
__status__ = "Development"

import sys
from locomotionbench.environment import *
from locomotionbench import cap, cop, fpe, zmp, com, base_orientation_err, impact, distance, foot_contact_velocity

# ARG1 = 'conf/experiment.yaml'
# ARG2 = 'conf/robot.yaml'

# TODO: Specify output dir in experiment.yaml
OUTPUT = 'output/test/'

USAGE = """usage: i3sa robot_model.lua joint_states.csv joint_velocities.csv joint_accelerations.csv joint_torques.csv grf_left.csv grf_right.csv conditions.yaml output_dir
robot_model.lua
joint_states.csv: TOBEDEFINED
joint_velocities.csv: TOBEDEFINED
joint_accelerations.csv: TOBEDEFINED
joint_torques.csv: TOBEDEFINED
grf_left.csv
grf_right.csv
conditions.yaml
"""
if __name__ == '__main__':
    Color.cyan_print("I3SA PI computation")

    # arg_len = 9
    # if len(sys.argv) != arg_len:
    #     print(USAGE)
    #     sys.exit(-1)

    temp_argv = ['conf/robot.yaml', 'input/2021_02_19/14/1/pos.csv', 'input/2021_02_19/14/1/vel.csv', 'input/2021_02_19/14/1/acc.csv', 'input/2021_02_19/14/1/trq.csv', 'input/2021_02_19/14/1/ftl.csv', 'input/2021_02_19/14/1/ftr.csv']
    #  model_path, pos_path, vel_path, acc_path, trq_path, grf_l_path, grf_r_path, conditions_path, output_folder_path = sys.argv[1:]
    output_folder_path = OUTPUT
    robot = Robot(temp_argv[0])
    experiment = Experiment(temp_argv[1:], robot.body_map)

    Color.green_print('Running Gait Phase Classification')
    robot.gait_segmentation(experiment, remove_ds=True, remove_hs=True)
    robot.create_contacts(experiment)

    Color.green_print("Running PI Center of Pressure")
    cop = cop.Cop(output_folder_path, robot=robot)
    is_ok = cop.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Capture Point")
    cap = cap.Cap(output_folder_path, robot=robot, experiment=experiment)
    is_ok = cap.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Foot Placement Estimator")
    fpe = fpe.Fpe(output_folder_path, robot=robot, experiment=experiment)
    is_ok = fpe.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Zero Moment Point")
    zmp = zmp.Zmp(output_folder_path, robot=robot, experiment=experiment)
    is_ok = zmp.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Center of Mass")
    com = com.Com(output_folder_path, robot=robot, experiment=experiment)
    com.performance_indicator()
    is_ok = com.loc()
    if not is_ok == 0:
        sys.exit(is_ok)
    is_ok = com.vel()
    if not is_ok == 0:
        sys.exit(is_ok)
    is_ok = com.acc()
    if not is_ok == 0:
        sys.exit(is_ok)
    is_ok = com.ang_mom()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Base Orientation Error")
    base_err = base_orientation_err.BaseOrientationError(output_folder_path, robot=robot, experiment=experiment)
    is_ok = base_err.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Impact")
    impact = impact.Impact(output_folder_path, robot=robot)
    is_ok = impact.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Distance Travelled")
    distance = distance.DistanceTravelled(output_folder_path, robot=robot, experiment=experiment)
    is_ok = distance.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    Color.green_print("Running PI Foot Contact Velocities")
    fc_vel = foot_contact_velocity.FootContactVelocity(output_folder_path, robot=robot, experiment=experiment)
    is_ok = fc_vel.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)
