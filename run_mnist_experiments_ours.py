import torch
import numpy as np
import os
from methods.mnist_x_model_selection_method import MNISTXModelSelectionMethod
from methods.mnist_xz_model_selection_method import MNISTXZModelSelectionMethod
from methods.mnist_z_model_selection_method import MNISTZModelSelectionMethod
from scenarios.abstract_scenario import AbstractScenario


SCENARIOS_NAMES = ["mnist_x", "mnist_z", "mnist_xz"]
SCENARIO_METHOD_CLASSES = {
    "mnist_x": MNISTXModelSelectionMethod,
    "mnist_z": MNISTZModelSelectionMethod,
    "mnist_xz": MNISTXZModelSelectionMethod,
}

RESULTS_FOLDER = "results/mnist/"


def run_experiment(scenario_name):
    # set random seed
    seed = 527
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_reps = 10

    print("\nLoading " + scenario_name + "...")
    scenario = AbstractScenario(filename="data/" + scenario_name + "/main.npz")
    scenario.info()
    scenario.to_tensor()
    scenario.to_cuda()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    for rep in range(num_reps):

        method_class = SCENARIO_METHOD_CLASSES[scenario_name]
        method = method_class(enable_cuda=torch.cuda.is_available())

        method.fit(train.x, train.z, train.y, dev.x, dev.z, dev.y,
                   g_dev=dev.g, verbose=True)
        g_pred_test = method.predict(test.x)
        mse = float(((g_pred_test - test.g) ** 2).mean())

        print("---------------")
        print("finished running methodology on scenario %s" % scenario)
        print("MSE on test:", mse)
        print("")
        print("saving results...")
        folder = "results/mnist/" + scenario_name + "/"
        file_name = "Ours_%d.npz" % rep
        save_path = os.path.join(folder, file_name)
        os.makedirs(folder, exist_ok=True)
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g,
                 g_hat=g_pred_test.detach())


def main():
    for scenario in SCENARIOS_NAMES:
        run_experiment(scenario)


if __name__ == "__main__":
    main()
