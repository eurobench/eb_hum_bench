#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_i3sa.py:
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

from src.locomotionbench.environment import *
from src.locomotionbench import cop, fpe, zmp, impact, distance, foot_contact_velocity
from src.locomotionbench import cap, base_orientation_err, com

USAGE = """usage: python3 src/script/run_i3sa.py tests/reemc_conf/robot.yaml tests/input/2021_02_19/14/1/pos.csv tests/input/2021_02_19/14/1/vel.csv tests/input/2021_02_19/14/1/acc.csv tests/input/2021_02_19/14/1/trq.csv tests/input/2021_02_19/14/1/ftl.csv tests/input/2021_02_19/14/1/ftr.csv tests/output/
robot.yaml: robot specific data
joint_states.csv: joint angle file
joint_velocities.csv: joint velocity file (optional)
joint_accelerations.csv: joint acceleration file (optional)
joint_torques.csv: joint torque file (optional)
grf_left.csv: ground reaction forces file of left foot (optional)
grf_right.csv: ground reaction forces file of right foot (optional)
conditions.yaml: TO BE DEFINED
"""
if __name__ == '__main__':
    Color.cyan_print("I3SA PI computation")

    #  temporary argument list for debugging IDE
    # temp_argv = ['src/script/run_i3sa.py', 'reemc_conf/robot.yaml', 'input/2021_02_19/14/1/pos.csv', 'input/2021_02_19/14/1/vel.csv',
    #              'input/2021_02_19/14/1/acc.csv', 'input/2021_02_19/14/1/trq.csv', 'input/2021_02_19/14/1/ftl.csv',
    #              'input/2021_02_19/14/1/ftr.csv', 'tests/output/']

    output_folder_path = sys.argv[-1]

    #  create experiment containing the uploaded files
    experiment = Experiment(sys.argv[2:-1])

    #  create robot with specific parameters used with experiment
    robot = Robot(sys.argv[1])

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