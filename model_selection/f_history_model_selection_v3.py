import itertools
import torch
import torch.nn as nn
from game_objectives.approximate_psi_objective import max_approx_psi_eval
from model_selection.learning_eval import f_history_g_eval


class FHistoryModelSelectionV3(object):
    def __init__(self, g_model_list, f_model_list, learning_args_list,
                 default_g_optimizer_factory, default_f_optimizer_factory,
                 g_simple_model_eval, f_simple_model_eval, learning_eval,
                 psi_eval_max_no_progress, psi_eval_burn_in):
        self.g_model_list = g_model_list
        self.f_model_list = f_model_list
        self.learning_args_list = learning_args_list

        self.default_g_optimizer_factory = default_g_optimizer_factory
        self.default_f_optimizer_factory = default_f_optimizer_factory

        self.g_simple_model_eval = g_simple_model_eval
        self.f_simple_model_eval = f_simple_model_eval
        self.learning_eval = learning_eval

        self.psi_eval_max_no_progress = psi_eval_max_no_progress
        self.psi_eval_burn_in = psi_eval_burn_in

    def do_model_selection(self, x_train, z_train, y_train,
                           x_dev, z_dev, y_dev, verbose=False):

        # first run learning evaluation on each hyperparameter setup
        f_of_z_dev_list = []
        e_dev_collections = []
        g_f_args_list = list(itertools.product(
            self.g_model_list, self.f_model_list, self.learning_args_list))

        for i, (g, f, learning_args) in enumerate(g_f_args_list):
            if verbose:
                print("starting learning args eval %d" % i)
            g.initialize()
            f.initialize()
            g_optimizer = learning_args["g_optimizer_factory"](g)
            f_optimizer = learning_args["f_optimizer_factory"](f)
            game_objective = learning_args["game_objective"]
            e_dev_list, f_of_z_dev_list = self.learning_eval.eval(
                x_train=x_train, z_train=z_train, y_train=y_train,
                x_dev=x_dev, z_dev=z_dev, y_dev=y_dev,
                g=g, f=f, g_optimizer=g_optimizer, f_optimizer=f_optimizer,
                game_objective=game_objective)
            f_of_z_dev_list.extend(f_of_z_dev_list)
            e_dev_collections.append(e_dev_list)

        # now find best hyperparameter setup based on saved parameters
        best_learning_args = None
        best_f_model = None
        best_g_model = None
        best_e_dev_tilde = None
        max_learning_eval = float("-inf")
        e_dev_tilde_list = [torch.stack(e_dev_list).mean(0) for
                            e_dev_list in e_dev_collections]
        e_dev_tilde = torch.stack(e_dev_tilde_list).mean(0)
        for i, (g, f, learning_args) in enumerate(g_f_args_list):
            e_dev_list = e_dev_collections[i]
            learning_eval, current_e_dev_tilde = max_approx_psi_eval(
                e_dev_list, f_of_z_dev_list, e_dev_tilde,
                max_no_progress=self.psi_eval_max_no_progress,
                burn_in=self.psi_eval_burn_in)
            if verbose:
                print("learning eval:", learning_eval,
                      learning_args["game_objective"],
                      learning_args["g_optimizer_factory"],
                      learning_args["f_optimizer_factory"])
            if learning_eval > max_learning_eval:
                max_learning_eval = learning_eval
                best_g_model = g
                best_f_model = f
                best_learning_args = learning_args
                best_e_dev_tilde = current_e_dev_tilde

        best_g_model.initialize()
        best_f_model.initialize()
        if verbose:
            print("size of f_z collection:", len(f_of_z_dev_list))
        return (best_g_model, best_f_model, best_learning_args,
                f_of_z_dev_list, best_e_dev_tilde)
