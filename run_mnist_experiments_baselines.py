import torch
import numpy as np
from baselines import all_baselines
from baselines.all_baselines import Poly2SLS, Vanilla2SLS, DirectNN, \
    DirectMNIST, GMM
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

    scenario = AbstractScenario(filename="data/" + scenario_name + "/main.npz")
    scenario.to_2d()
    scenario.info()

    train = scenario.get_dataset("train")
    dev = scenario.get_dataset("dev")
    test = scenario.get_dataset("test")

    for rep in range(num_reps):
        # Not all methods are applicable in all scenarios

        methods = []

        # baseline methods
        poly2sls_method = Poly2SLS(poly_degree=[1],
                                   ridge_alpha=np.logspace(-5, 3, 5))
        methods += [("Ridge2SLS", poly2sls_method)]
        methods += [("Vanilla2SLS", Vanilla2SLS())]
        direct_method = None
        gmm_method = None
        methods += [("DeepIV", all_baselines.DeepIV())]
        if scenario_name == "mnist_z":
            methods += [("DeepIV", all_baselines.DeepIV(treatment_model="cnn"))]
            gmm_method = GMM(
                g_model="2-layer", n_steps=10, g_epochs=10)
            direct_method = DirectNN()
        elif scenario_name == "mnist_x":
            gmm_method = GMM(g_model="mnist", n_steps=10, g_epochs=1)
            direct_method = DirectMNIST()
        elif scenario_name == "mnist_xz":
            gmm_method = GMM(g_model="mnist", n_steps=10, g_epochs=1)
            direct_method = DirectMNIST()
        methods += [("DirectNN", direct_method)]
        methods += [("GMM", gmm_method)]

        for method_name, method in methods:
            print("Running " + method_name)
            model = method.fit(train.x, train.y, train.z, None)
            folder = "results/mnist/" + scenario_name + "/"
            file_name = "%s_%d.npz" % (method_name, rep)
            save_path = os.path.join(folder, file_name)
            os.makedirs(folder, exist_ok=True)
            save_model(model, save_path, test)
            test_mse = eval_model(model, test)
            model_type_name = type(model).__name__
            print("Test MSE of %s: %f" % (model_type_name, test_mse))


def main():
    # scenarios = ["mnist_z", "mnist_x", "mnist_xz"]
    scenarios = ["mnist_xz"]
    for scenario in scenarios:
        print("\nLoading " + scenario + "...")
        run_experiment(scenario)


if __name__ == "__main__":
    main()

