import yaml
from locomotionbench import utility, metrics

# TODO: Use arguments from cli. not implemented for debugging purposes

ARG1 = 'conf/experiment.yaml'
ARG2 = 'conf/robot.yaml'

if __name__ == '__main__':

    e_yaml = open(ARG1)
    r_yaml = open(ARG2)

    experiment = yaml.load(e_yaml, Loader=yaml.FullLoader)
    robot = yaml.load(r_yaml, Loader=yaml.FullLoader)

    exp = metrics.Metrics(robot, experiment)
    metrics = exp.calc_metrics()
    distance_traveled, n_steps, normalized_dist_steps = exp.get_n_steps_normalized_by_leg_distance()
    impact = exp.get_impact()
    print('---')
