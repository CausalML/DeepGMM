from scenarios.toy_scenarios import HingeLinearScenario, HeaviSideScenario, Zoo, \
    Standardizer, AGMMZoo
import numpy as np


def create_dataset(function_name='step', dir="data/zoo/"):
    # set random seed
    seed = 527
    np.random.seed(seed)

    # set up model classes, objective, and data scenario
    num_train = 2000
    num_dev = 2000
    num_test = 2000

    scenario = Standardizer(
        AGMMZoo(function_name, two_gps=False, n_instruments=2))
    scenario.setup(num_train=num_train, num_dev=num_dev, num_test=num_test)
    scenario.to_file(dir + function_name)


if __name__ == "__main__":
    for function in ['linear', 'abs', 'sin', 'step']:
        create_dataset(function)
