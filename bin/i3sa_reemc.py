import yaml
from locomotionbench import utility, metrics, indicators
import matplotlib.pyplot as plt

# TODO: Use arguments from cli. not implemented for debugging purposes

ARG1 = 'conf/experiment.yaml'
ARG2 = 'conf/robot.yaml'
OUTPUT = 'output/'
if __name__ == '__main__':

    e_yaml = open(ARG1)
    r_yaml = open(ARG2)
    export = utility.PIExporter()
    experiment = yaml.load(e_yaml, Loader=yaml.FullLoader)
    robot = yaml.load(r_yaml, Loader=yaml.FullLoader)

    exp = metrics.Metrics(robot, experiment)
    gait_phases = exp.get_gait_segments()
    metrics = exp.calc_metrics(gait_phases)

    distance_traveled, n_steps, normalized_dist_steps = exp.get_n_steps_normalized_by_leg_distance(gait_phases)
    impact = exp.get_impact(gait_phases)

    indicators = indicators.Indicators(experiment, metrics)
    indicators.ping()

    # export.export_scalar(impact, OUTPUT + 'impact.yaml')
    # export.export_scalar(distance_traveled, OUTPUT + 'distance_traveled.yaml')
    # export.export_scalar(n_steps, OUTPUT + 'n_steps.yaml')
    # export.export_scalar(normalized_dist_steps, OUTPUT + 'normalized_dist_steps.yaml')

    print('---')
