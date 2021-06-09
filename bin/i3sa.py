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

from locomotionbench.environment import *
from locomotionbench import cap, cop, fpe, zmp, com, base_orientation_err, impact, distance, foot_contact_velocity

# ARG1 = 'conf/experiment.yaml'
# ARG2 = 'conf/robot.yaml'

# TODO: Specify output dir in experiment.yaml
OUTPUT = 'output/test/'

USAGE = """usage: i3sa robot_model.lua joint_states.csv joint_velocities.csv joint_accelerations.csv joint_torques.csv grf_left.csv grf_right.csv conditions.yaml output_dir
robot_model.lua
joint_states.csv: TO BE DEFINED
joint_velocities.csv: TO BE DEFINED
joint_accelerations.csv: TO BE DEFINED
joint_torques.csv: TO BE DEFINED
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

    #  temporary argument list for debugging IDE
    temp_argv = ['conf/robot.yaml', 'input/2021_02_19/14/1/pos.csv', 'input/2021_02_19/14/1/vel.csv',
                 'input/2021_02_19/14/1/acc.csv', 'input/2021_02_19/14/1/trq.csv', 'input/2021_02_19/14/1/ftl.csv',
                 'input/2021_02_19/14/1/ftr.csv']
    #  model_path, pos_path, vel_path, acc_path, trq_path, grf_l_path, grf_r_path, conditions_path, output_folder_path = sys.argv[1:]

    # temporary static output folder path for debugging in IDE
    output_folder_path = OUTPUT

    #  create experiment containing the uploaded files
    experiment = Experiment(temp_argv[1:])

    #  create robot with specific parameters used with experiment
    robot = Robot(temp_argv[0])

    #  check if column order in file header does match link order in robot model.
    if experiment.are_columns_matching(robot.get_body_map()):
        #  reindex otherwise
        experiment.reindex_columns(robot.get_body_map())

    Color.green_print('Running Gait Phase Classification')
    #  classify gait phases to single left, single right and double support phases.
    #  possibility to remove first/last double support phase and additionally first/last halfstep leading into the motion
    truncate = robot.gait_segmentation(experiment, remove_ds=True, remove_hs=True)
    if all(truncate):
        experiment.truncate_lead_time(truncate)
        experiment.truncate_data(truncate)
        robot.truncate_phases(truncate)
        robot.truncate_step_list(truncate)

    #  create foot contacts containing all meta information of the contact for each of the different contact phases
    robot.create_contacts(experiment)

    ############################################################################
    #  Start of function calls for all the individual performance indicators   #
    ############################################################################
    #  Each PI uses the robot and experiment class to load necessary files.    #
    #  The exact REQUIRED files are listed in each class and loaded applicable #
    ############################################################################

    Color.green_print("Running PI Center of Pressure")
    try:
        cop = cop.Cop(output_folder_path, robot=robot, experiment=experiment)
        is_ok = cop.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Capture Point")
    try:
        cap = cap.Cap(output_folder_path, robot=robot, experiment=experiment)
        is_ok = cap.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Foot Placement Estimator")
    try:
        fpe = fpe.Fpe(output_folder_path, robot=robot, experiment=experiment)
        is_ok = fpe.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Zero Moment Point")
    try:
        zmp = zmp.Zmp(output_folder_path, robot=robot, experiment=experiment)
        is_ok = zmp.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Center of Mass")
    try:
        com = com.Com(output_folder_path, robot=robot, experiment=experiment)
        is_ok = com.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Base Orientation Error")
    try:
        base_err = base_orientation_err.BaseOrientationError(output_folder_path, robot=robot, experiment=experiment)
        is_ok = base_err.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Impact")
    try:
        impact = impact.Impact(output_folder_path, robot=robot, experiment=experiment)
        is_ok = impact.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Distance Travelled")
    try:
        distance = distance.DistanceTravelled(output_folder_path, robot=robot, experiment=experiment)
        is_ok = distance.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Foot Contact Velocities")
    try:
        fc_vel = foot_contact_velocity.FootContactVelocity(output_folder_path, robot=robot, experiment=experiment)
        is_ok = fc_vel.performance_indicator()
        if not is_ok == 0:
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')
