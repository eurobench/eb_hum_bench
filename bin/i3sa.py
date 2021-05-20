import sys
from locomotionbench.utility import IOHandler
from locomotionbench.experiment_factory import ExperimentFactory
from locomotionbench.metrics import Metrics
from locomotionbench import cap, cop, fpe, zmp, com, base_orientation_err
from locomotionbench.environment import Robot, Experiment
from locomotionbench.gait_segmentation import *
from locomotionbench.indicators import Indicators
import matplotlib.pyplot as plt
import timeit

# ARG1 = 'conf/experiment.yaml'
# ARG2 = 'conf/robot.yaml'

# TODO: Specify output dir in experiment.yaml
OUTPUT = 'output/'

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
    print("I3SA PI computation")

    # arg_len = 9
    # if len(sys.argv) != arg_len:
    #     print(USAGE)
    #     sys.exit(-1)

    temp_argv = ['conf/robot.yaml', 'input/2021_02_19/14/1/pos.csv', 'input/2021_02_19/14/1/vel.csv', 'input/2021_02_19/14/1/acc.csv', 'input/2021_02_19/14/1/trq.csv', 'input/2021_02_19/14/1/ftl.csv', 'input/2021_02_19/14/1/ftr.csv']
    #model_path, pos_path, vel_path, acc_path, trq_path, grf_l_path, grf_r_path, conditions_path, output_folder_path = sys.argv[1:]
    output_folder_path = OUTPUT
    robot = Robot(temp_argv[0])
    experiment = Experiment(temp_argv[1:], robot.body_map)

    print("Running Gait Phase Classification")
    robot.gait_segmentation(experiment, remove_ds=True, remove_hs=True)
    robot.create_contacts(experiment)

    print("Running PI Center of Pressure")
    require = ['phases', 'cos']
    cop = cop.Cop(require, output_folder_path, robot, experiment)
    is_ok = cop.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    print("Running PI Capture Point")
    require = ['pos', 'vel', 'cos']
    cap = cap.Cap(require, output_folder_path, robot, experiment, True)
    is_ok = cap.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    print("Running PI Foot Placement Estimator")
    require = ['pos', 'vel', 'cos']
    fpe = fpe.Fpe(require, output_folder_path, robot, experiment, True)
    is_ok = fpe.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    print("Running PI Zero Moment Point")
    require = ['pos', 'vel', 'acc']
    zmp = zmp.Zmp(require, output_folder_path, robot, experiment)
    is_ok = zmp.performance_indicator()
    if not is_ok == 0:
        sys.exit(is_ok)

    print("Running PI Center of Mass")
    require = ['pos', 'vel', 'acc']
    com = com.Com(require, output_folder_path, robot, experiment)
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

    print("Running PI Base Orientation Error")
    require = ['pos', 'phases']
    base_err = base_orientation_err.BaseOrientationError(require, output_folder_path, robot, experiment)
    is_ok = base_err.performance_indicator()

    # #  Initialize experiment: load robot.yaml and experiment.yaml
    # robot, experiment = IOHandler.init(sys.argv[1:])
    # #  Create instances of metrics according to several difficulty settings of the test and testing repetitions
    # # factory = ExperimentFactory(experiment, robot)
    #
    # #  If no factory is used create metrics for just one experiment (one difficulty setting, one repetition)
    # exp = Metrics(robot, experiment)
    #
    # #  Calculate gait phases. Time series information will be normalized according to the individual phases
    # gait_phases = exp.get_gait_segments()
    #
    # #  Create dataframe containing information of all metrics for each frame which will be used for normalization wrt gait phases
    # metrics = exp.calc_metrics(gait_phases)
    #
    # #  High level indicators which are not needed to be normalized
    # distance_traveled, n_steps, normalized_dist_steps = exp.get_n_steps_normalized_by_leg_distance(gait_phases)
    # impact = exp.get_impact(gait_phases)
    #
    # #  Class which will use metrics dataframe for aggregation of data (min, max, mean, std ...)
    #
    # #  TODO: Create normalization/aggregation for PI output wrt the following nesting:
    # """
    # 1. One executed protocol with multiple trials according to the different difficulty settings (e.g. step length, stair height ...)
    # 2. With each trial containing multiple repetitions with the same difficulty setting
    # 3. Each containing several gait phases which are the basis for normalization
    # """
    # #  TODO: ---
    # # indicators = Indicators(experiment, metrics, gait_phases)
    # # indicators.get_single_phases_per_ptype('fl_single')
    #
    # #  Output calculated metrics to .yaml format
    # IOHandler.export_scalar(impact, OUTPUT + 'impact.yaml')
    # print('File ', OUTPUT + 'impact.yaml', ' written')
    # IOHandler.export_scalar(distance_traveled, OUTPUT + 'distance_traveled.yaml')
    # print('File ', OUTPUT + 'distance_traveled.yaml', ' written')
    # IOHandler.export_scalar(n_steps, OUTPUT + 'n_steps.yaml')
    # print('File ', OUTPUT + 'n_steps.yaml', ' written')
    # IOHandler.export_scalar(normalized_dist_steps, OUTPUT + 'normalized_dist_steps.yaml')
    # print('File ', OUTPUT + 'normalized_dist_steps.yaml', ' written')

