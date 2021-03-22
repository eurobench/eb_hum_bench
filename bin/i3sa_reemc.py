import yaml
from locomotionbench import utility, metrics
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
    # plt.plot(exp.lead_time, exp.ftl['force_z'])
    # plt.plot(exp.lead_time, exp.ftr['force_z'])
    exp.crop_start_end_phases()
    metrics = exp.calc_metrics()
    # plt.show()
    # distance_traveled, n_steps, normalized_dist_steps = exp.get_n_steps_normalized_by_leg_distance()
    # impact = exp.get_impact()

    # export.export_scalar(impact, OUTPUT + 'impact.yaml')
    # export.export_scalar(distance_traveled, OUTPUT + 'distance_traveled.yaml')
    # export.export_scalar(n_steps, OUTPUT + 'n_steps.yaml')
    # export.export_scalar(normalized_dist_steps, OUTPUT + 'normalized_dist_steps.yaml')

    print('---')
