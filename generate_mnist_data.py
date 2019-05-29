import numpy as np
import torch

from scenarios.mnist_scenarios import MNISTScenarioZ, MNISTScenarioX, \
    MNISTScenarioXZ
from scenarios.toy_scenarios import Standardizer


def create_dataset(scenario_class, dir):
    # set random seed
    seed = 527
    np.random.seed(seed)
    torch.manual_seed(seed)

    # set up model classes, objective, and data scenario
    num_train = 20000
    num_dev = 10000
    num_test = 10000
    scenario = Standardizer(scenario_class(g_function="abs"))

    scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)
    scenario.info()
    scenario.to_file(dir)


if __name__ == "__main__":
    for scenario, path in [(MNISTScenarioX, "mnist_x"),
                           (MNISTScenarioZ, "mnist_z"),
                           (MNISTScenarioXZ, "mnist_xz")]:
        print("Creating " + path + " ...")
        create_dataset(scenario, "data/" + path + "/main")
