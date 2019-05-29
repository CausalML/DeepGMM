import torch
import torch.nn as nn
from model_selection.learning_eval import f_history_g_eval


class FHistoryModelSelectionV1(object):
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
                           x_dev, z_dev, y_dev):
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
            print("g eval: %f ± %f" % (g_eval, g_eval_std))
        i_max = max(range(num_g), key=lambda i_: g_model_eval[i_])
        penalty = self.gamma * g_model_eval_std[i_max]
        opt_list = [i for i in range(i_max + 1) if
                    g_model_eval[i] >= g_model_eval[i_max] - penalty]
        i_opt = opt_list[0]
        print("optimal g model index:", i_opt)
        best_g_model = self.g_model_list[i_opt]

        # second select f model
        e_train = y_train - best_g_model(x_train).detach()
        e_dev = y_dev - best_g_model(x_dev).detach()
        f_model_eval = []
        f_model_eval_std = []
        num_f = len(self.f_model_list)
        for f_model in self.f_model_list:
            f_model.initialize()
            optimizer = self.default_f_optimizer_factory(f_model)
            f_eval, f_eval_std = self.f_simple_model_eval.eval(
                f=f_model, f_optimizer=optimizer, x_train=z_train, x_dev=z_dev,
                y_train=e_train, y_dev=e_dev)
            f_model_eval.append(f_eval)
            f_model_eval_std.append(f_eval_std)
            print("f eval: %f ± %f" % (f_eval, f_eval_std))
        i_max = max(range(num_f), key=lambda i_: f_model_eval[i_])
        penalty = self.gamma * f_model_eval_std[i_max]
        opt_list = [i for i in range(i_max + 1) if
                    f_model_eval[i] >= f_model_eval[i_max] - penalty]
        i_opt = opt_list[0]
        print("optimal f model index:", i_opt)
        best_f_model = self.f_model_list[i_opt]

        # third, select best learning args
        f_of_z_dev_collection = []
        e_dev_list = []
        for i, learning_args in enumerate(self.learning_args_list):
            print("starting learning args eval %d" % i)
            best_g_model.initialize()
            best_f_model.initialize()
            g_optimizer = learning_args["g_optimizer_factory"](best_g_model)
            f_optimizer = learning_args["f_optimizer_factory"](best_f_model)
            game_objective = learning_args["game_objective"]
            _, e_dev, f_of_z_dev_list = self.learning_eval.eval(
                x_train=x_train, z_train=z_train, y_train=y_train,
                x_dev=x_dev, z_dev=z_dev, y_dev=y_dev,
                g=best_g_model, f=best_f_model,
                g_optimizer=g_optimizer, f_optimizer=f_optimizer,
                game_objective=game_objective)
            f_of_z_dev_collection.extend(f_of_z_dev_list)
            e_dev_list.append(e_dev)

        best_learning_args = None
        max_learning_eval = float("-inf")
        for learning_args, e_dev in zip(self.learning_args_list, e_dev_list):
            # print(e_dev.shape)
            # print(f_of_z_dev_collection[0].shape)
            learning_eval = f_history_g_eval(e_dev, f_of_z_dev_collection)
            print("learning eval:", learning_eval,
                  learning_args["game_objective"],
                  learning_args["g_optimizer_factory"],
                  learning_args["f_optimizer_factory"])
            if learning_eval > max_learning_eval:
                max_learning_eval = learning_eval
                best_learning_args = learning_args
                print("update")

        best_g_model.initialize()
        best_f_model.initialize()
        return (best_g_model, best_f_model,
                best_learning_args, f_of_z_dev_collection)
