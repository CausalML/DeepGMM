import torch
import numpy as np
import os
from scenarios.abstract_scenario import AbstractScenario
from methods.toy_model_selection_method import ToyModelSelectionMethod


def run_experiment(scenario_name):
    # set random seed
    seed = 527
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_reps = 10

    print("\nLoading " + scenario_name + "...")
    scenario = AbstractScenario(filename="data/zoo/" + scenario_name + ".npz")
    scenario.info()
    scenario.to_tensor()
    scenario.to_cuda()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    for rep in range(num_reps):

        method = ToyModelSelectionMethod(enable_cuda=torch.cuda.is_available())
        method.fit(train.x, train.z, train.y, dev.x, dev.z, dev.y,
                   g_dev=dev.g, verbose=True)
        g_pred_test = method.predict(test.x)
        mse = float(((g_pred_test - test.g) ** 2).mean())

        print("---------------")
        print("finished running methodology on scenario %s" % scenario)
        print("MSE on test:", mse)
        print("")
        print("saving results...")
        folder = "results/zoo/" + scenario_name + "/"
        file_name = "Ours_%d.npz" % rep
        save_path = os.path.join(folder, file_name)
        os.makedirs(folder, exist_ok=True)
        np.savez(save_path, x=test.w, y=test.y, g_true=test.g,
                 g_hat=g_pred_test.detach())


def main():
    for scenario in ["step", "sin", "abs", "linear"]:
        run_experiment(scenario)
    # run_experiment("linear")


if __name__ == "__main__":
    main()
