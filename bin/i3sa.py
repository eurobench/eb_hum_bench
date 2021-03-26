import sys
from locomotionbench.utility import IOHandler
from locomotionbench.experiment_factory import ExperimentFactory
from locomotionbench.metrics import Metrics
from locomotionbench.indicators import Indicators
import matplotlib.pyplot as plt

# ARG1 = 'conf/experiment.yaml'
# ARG2 = 'conf/robot.yaml'

# TODO: Specify output dir in experiment.yaml
OUTPUT = 'output/'

if __name__ == '__main__':

    #  Initialize experiment: load robot.yaml and experiment.yaml
    robot, experiment = IOHandler.init(sys.argv[1:])
    #  Create instances of metrics according to several difficulty settings of the test and testing repetitions
    # factory = ExperimentFactory(experiment, robot)

    #  If no factory is used create metrics for just one experiment (one difficulty setting, one repetition)
    exp = Metrics(robot, experiment)

    #  Calculate gait phases. Time series information will be normalized according to the individual phases
    gait_phases = exp.get_gait_segments()

    #  Create dataframe containing information of all metrics for each frame which will be used for normalization wrt gait phases
    metrics = exp.calc_metrics(gait_phases)

    #  High level indicators which are not needed to be normalized
    distance_traveled, n_steps, normalized_dist_steps = exp.get_n_steps_normalized_by_leg_distance(gait_phases)
    impact = exp.get_impact(gait_phases)

    #  Class which will use metrics dataframe for aggregation of data (min, max, mean, std ...)

    #  TODO: Create normalization/aggregation for PI output wrt the following nesting:
    """
    1. One executed protocol with multiple trials according to the different difficulty settings (e.g. step length, stair height ...)
    2. With each trial containing multiple repetitions with the same difficulty setting
    3. Each containing several gait phases which are the basis for normalization
    """
    #  TODO: ---
    # indicators = Indicators(experiment, metrics, gait_phases)
    # indicators.get_single_phases_per_ptype('fl_single')

    #  Output calculated metrics to .yaml format
    IOHandler.export_scalar(impact, OUTPUT + 'impact.yaml')
    print('File ', OUTPUT + 'impact.yaml', ' written')
    IOHandler.export_scalar(distance_traveled, OUTPUT + 'distance_traveled.yaml')
    print('File ', OUTPUT + 'distance_traveled.yaml', ' written')
    IOHandler.export_scalar(n_steps, OUTPUT + 'n_steps.yaml')
    print('File ', OUTPUT + 'n_steps.yaml', ' written')
    IOHandler.export_scalar(normalized_dist_steps, OUTPUT + 'normalized_dist_steps.yaml')
    print('File ', OUTPUT + 'normalized_dist_steps.yaml', ' written')

