import itertools
import torch
import torch.nn as nn
from model_selection.learning_eval import f_history_g_eval


class FHistoryModelSelectionV2(object):
    def __init__(self, g_model_list, f_model_list, learning_args_list,
                 default_g_optimizer_factory, default_f_optimizer_factory,
                 g_simple_model_eval, f_simple_model_eval, learning_eval,
                 gamma):
        self.g_model_list = g_model_list
        self.f_model_list = f_model_list
        self.learning_args_list = learning_args_list

        self.default_g_optimizer_factory = default_g_optimizer_factory
        self.default_f_optimizer_factory = default_f_optimizer_factory

        self.g_simple_model_eval = g_simple_model_eval
        self.f_simple_model_eval = f_simple_model_eval
        self.learning_eval = learning_eval

        self.gamma = gamma

    def do_model_selection(self, x_train, z_train, y_train,
                           x_dev, z_dev, y_dev, verbose=False):
        # first select g model
        g_model_eval = []
        g_model_eval_std = []
        num_g = len(self.g_model_list)
        for g_model in self.g_model_list:
            g_model.initialize()
            optimizer = self.default_g_optimizer_factory(g_model)
            g_eval, g_eval_std = self.g_simple_model_eval.eval(
                f=g_model, f_optimizer=optimizer, x_train=x_train, x_dev=x_dev,
                y_train=y_train, y_dev=y_dev)
            g_model_eval.append(g_eval)
            g_model_eval_std.append(g_eval_std)
            if verbose:
                print("g eval: %f ± %f" % (g_eval, g_eval_std))
        i_max = max(range(num_g), key=lambda i_: g_model_eval[i_])
        penalty = self.gamma * g_model_eval_std[i_max]
        opt_list = [i for i in range(i_max + 1) if
                    g_model_eval[i] >= g_model_eval[i_max] - penalty]
        i_opt = opt_list[0]
        if verbose:
            print("optimal g model index:", i_opt)
        best_g_model = self.g_model_list[i_opt]

        # second, select best learning args and f function
        f_of_z_dev_collection = []
        e_dev_list = []
        f_args_list = list(itertools.product(
            self.f_model_list, self.learning_args_list))

        for i, (f, learning_args) in enumerate(f_args_list):
            if verbose:
                print("starting learning args eval %d" % i)
            best_g_model.initialize()
            f.initialize()
            g_optimizer = learning_args["g_optimizer_factory"](best_g_model)
            f_optimizer = learning_args["f_optimizer_factory"](f)
            game_objective = learning_args["game_objective"]
            _, e_dev, f_of_z_dev_list = self.learning_eval.eval(
                x_train=x_train, z_train=z_train, y_train=y_train,
                x_dev=x_dev, z_dev=z_dev, y_dev=y_dev,
                g=best_g_model, f=f, g_optimizer=g_optimizer, f_optimizer=f_optimizer,
                game_objective=game_objective)
            f_of_z_dev_collection.extend(f_of_z_dev_list)
            e_dev_list.append(e_dev)

        best_learning_args = None
        best_f_model = None
        max_learning_eval = float("-inf")
        for i, (f, learning_args) in enumerate(f_args_list):
            e_dev = e_dev_list[i]
            learning_eval = f_history_g_eval(e_dev, f_of_z_dev_collection)
            if verbose:
                print("learning eval:", learning_eval,
                      learning_args["game_objective"],
                      learning_args["g_optimizer_factory"],
                      learning_args["f_optimizer_factory"])
            if learning_eval > max_learning_eval:
                max_learning_eval = learning_eval
                best_f_model = f
                best_learning_args = learning_args

        best_g_model.initialize()
        best_f_model.initialize()
        if verbose:
            print("size of f_z collection:", len(f_of_z_dev_collection))
        return (best_g_model, best_f_model,
                best_learning_args, f_of_z_dev_collection)
