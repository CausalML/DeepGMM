import torch
import numpy as np
from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
    GMM, DeepIV, AGMM
import os
from scenarios.abstract_scenario import AbstractScenario
import tensorflow


def eval_model(model, test):
    g_pred_test = model.predict(test.x)
    mse = float(((g_pred_test - test.g) ** 2).mean())
    return mse


def save_model(model, save_path, test):
    g_pred = model.predict(test.x)
    np.savez(save_path, x=test.w, y=test.y, g_true=test.g, g_hat=g_pred)


def run_experiment(scenario_name, num_reps=10, seed=527):
    # set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    tensorflow.set_random_seed(seed)

    scenario_path = "data/zoo/" + scenario_name + ".npz"
    scenario = AbstractScenario(filename=scenario_path)
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    for rep in range(num_reps):
        # Not all methods are applicable in all scenarios

        methods = []

        # baseline methods
        methods += [("Poly2SLS", Poly2SLS())]
        methods += [("Vanilla2SLS", Vanilla2SLS())]
        methods += [("DirectNN", DirectNN())]
        methods += [("GMM", GMM(g_model="2-layer", n_steps=20))]
        methods += [("AGMMw", AGMM())]
        methods += [("DeepIV", DeepIV())]

        for method_name, method in methods:
            print("Running " + method_name)
            model = method.fit(train.x, train.y, train.z, None)
            folder = "results/zoo/" + scenario_name + "/"
            file_name = "%s_%d.npz" % (method_name, rep)
            save_path = os.path.join(folder, file_name)
            os.makedirs(folder, exist_ok=True)
            save_model(model, save_path, test)
            test_mse = eval_model(model, test)
            model_type_name = type(model).__name__
            print("Test MSE of %s: %f" % (model_type_name, test_mse))


def main():
    # scenarios = ["step", "sin", "abs", "linear"]
    scenarios = ["linear"]
    for scenario in scenarios:
        print("\nLoading " + scenario + "...")
        run_experiment(scenario)


if __name__ == "__main__":
    main()

