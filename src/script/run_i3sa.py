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
from src.locomotionbench import cop, fpe, zmp, impact, distance, foot_contact_velocity, grf, res_forces
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

def main(args):
    Color.cyan_print("I3SA PI computation")

    output_folder_path = args.out

    #  create experiment containing the uploaded files
    experiment = Experiment(args)

    #  create robot with specific parameters used with experiment
    robot = Robot(args)

    #  check if column order in file header does match link order in robot model.
    if experiment.are_columns_matching(robot.get_body_map()):
        #  reindex otherwise
        experiment.reindex_columns(robot.get_body_map())

    Color.green_print('Running Gait Phase Classification')
    #  classify gait phases to single left, single right and double support phases.
    #  possibility to remove first/last double support phase and additionally first/last halfstep leading into the motion
    robot.assign_phase(experiment.lead_time.to_numpy().flatten())
    truncate = robot.truncate_gait(remove_ds=True, remove_hs=True)
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
        my_cop = cop.Cop(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_cop.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Capture Point")
    try:
        my_cap = cap.Cap(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_cap.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Foot Placement Estimator")
    try:
        my_fpe = fpe.Fpe(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_fpe.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Zero Moment Point")
    try:
        my_zmp = zmp.Zmp(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_zmp.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Center of Mass")
    try:
        my_com = com.Com(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_com.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Base Orientation Error")
    try:
        base_err = base_orientation_err.BaseOrientationError(output_folder_path, robot=robot, experiment=experiment)
        is_ok = base_err.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Impact")
    try:
        my_impact = impact.Impact(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_impact.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Distance Travelled")
    try:
        my_distance = distance.DistanceTravelled(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_distance.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Foot Contact Velocities")
    try:
        fc_vel = foot_contact_velocity.FootContactVelocity(output_folder_path, robot=robot, experiment=experiment)
        is_ok = fc_vel.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Ground Reaction Forces")
    try:
        my_grf = grf.GroundReactionForces(output_folder_path, robot=robot, experiment=experiment)
        is_ok = my_grf.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')

    Color.green_print("Running PI Residual Forces")
    try:
        res = res_forces.ResidualForces(output_folder_path, robot=robot, experiment=experiment)
        is_ok = res.performance_indicator()
        if not is_ok == 0:
            Color.err_print('Not ok')
            sys.exit(is_ok)
    except FileNotFoundError:
        Color.warning_print('Skipping PI')


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    main(args)


ENTRY_USAGE = """
usage: run_i3sa tests/reemc_conf/robot.yaml tests/reemc_conf/reemc.lua tests/input/2021_02_19/14/1/pos.csv tests/input/2021_02_19/14/1/vel.csv tests/input/2021_02_19/14/1/acc.csv tests/input/2021_02_19/14/1/trq.csv tests/input/2021_02_19/14/1/ftl.csv tests/input/2021_02_19/14/1/ftr.csv tests/input/2021_02_19/14/1/gaitEvents.yaml output/
robot.yaml: robot specific data
robot.lua: robot description
joint_states.csv: joint angle file
joint_velocities.csv: joint velocity file (optional)
joint_accelerations.csv: joint acceleration file (optional)
joint_torques.csv: joint torque file (optional)
grf_left.csv: ground reaction forces file of left foot (optional)
grf_right.csv: ground reaction forces file of right foot (optional)
gaitEvents.yaml: contact information
"""


def entry_point():
    if len(sys.argv) != 11:
        Color.err_print('Wrong input parameters')
        Color.err_print(ENTRY_USAGE)
        sys.exit(1)
    file_conf = sys.argv[1]
    file_model = sys.argv[2]
    file_pos = sys.argv[3]
    file_vel = sys.argv[4]
    file_acc = sys.argv[5]
    file_trq = sys.argv[6]
    file_ftl = sys.argv[7]
    file_ftr = sys.argv[8]
    file_gait = sys.argv[9]
    dir_out = sys.argv[10]

    l_arg = ['--conf', file_conf]
    l_arg.extend(['--model', file_model])
    l_arg.extend(['--pos', file_pos])
    l_arg.extend(['--vel', file_vel])
    l_arg.extend(['--acc', file_acc])
    l_arg.extend(['--trq', file_trq])
    l_arg.extend(['--ftl', file_ftl])
    l_arg.extend(['--ftr', file_ftr])
    l_arg.extend(['--gait', file_gait])
    l_arg.extend(['--out', dir_out])

    args = parse_args(l_arg)
    sys.exit(main(args))
