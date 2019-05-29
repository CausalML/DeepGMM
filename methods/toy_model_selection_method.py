import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from game_objectives.simple_moment_objective import OptimalMomentObjective
from learning.learning_dev_f import GradientDescentLearningDevF, SGDLearningDevF

from methods.abstract_method import AbstractMethod
from model_selection.f_history_model_selection_v3 import \
    FHistoryModelSelectionV3
from model_selection.learning_eval import FHistoryLearningEvalGradientDecent
from model_selection.learning_eval_nostop import \
    FHistoryLearningEvalGradientDecentNoStop, FHistoryLearningEvalNoStop, \
    FHistoryLearningEvalSGDNoStop
from model_selection.simple_model_eval import GradientDecentSimpleModelEval, \
    SGDSimpleModelEval
from models.mlp_model import MLPModel
from optimizers import OAdam
from optimizers.optimizer_factory import OptimizerFactory


class ToyModelSelectionMethod(AbstractMethod):
    def __init__(self, enable_cuda=False):
        AbstractMethod.__init__(self)
        self.g = None
        self.f = None
        self.dev_f_collection = None

        g_models = [
            MLPModel(input_dim=1, layer_widths=[20, 3],
                     activation=nn.LeakyReLU).double(),
        ]
        f_models = [
            MLPModel(input_dim=2, layer_widths=[20],
                     activation=nn.LeakyReLU).double(),
        ]
        if torch.cuda.is_available() and enable_cuda:
            for i, g in enumerate(g_models):
                g_models[i] = g.cuda()
            for i, f in enumerate(f_models):
                f_models[i] = f.cuda()

        g_learning_rates = [0.0005, 0.0002, 0.0001]
        game_objectives = [
            OptimalMomentObjective(),
        ]
        # g_learning_rates = [0.0005]
        # game_objectives = [OptimalMomentObjective(lambda_1=0.5)]
        learning_setups = []
        for g_lr in g_learning_rates:
            for game_objective in game_objectives:
                learning_setup = {
                    "g_optimizer_factory": OptimizerFactory(
                        OAdam, lr=float(g_lr), betas=(0.5, 0.9)),
                    "f_optimizer_factory": OptimizerFactory(
                        OAdam, lr=5.0*float(g_lr), betas=(0.5, 0.9)),
                    "game_objective": game_objective
                }
                learning_setups.append(learning_setup)

        default_g_opt_factory = OptimizerFactory(
            Adam, lr=0.001, betas=(0.5, 0.9))
        default_f_opt_factory = OptimizerFactory(
            Adam, lr=0.005, betas=(0.5, 0.9))
        g_simple_model_eval = SGDSimpleModelEval()
        f_simple_model_eval = SGDSimpleModelEval()
        learning_eval = FHistoryLearningEvalSGDNoStop(
            num_epochs=3000, eval_freq=20, print_freq=100, batch_size=1024)
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=g_models,
            f_model_list=f_models,
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_max_no_progress=20, psi_eval_burn_in=50)
        self.default_g_opt_factory = default_g_opt_factory

    def fit(self, x_train, z_train, y_train, x_dev, z_dev, y_dev,
            video_plotter=None, verbose=False, g_dev=None):
        g, f, learning_args, dev_f_collection, e_dev_tilde = \
            self.model_selection.do_model_selection(
                x_train=x_train, z_train=z_train, y_train=y_train,
                x_dev=x_dev, z_dev=z_dev, y_dev=y_dev, verbose=verbose)
        self.g = g
        self.f = f
        self.dev_f_collection = dev_f_collection
        g_optimizer = learning_args["g_optimizer_factory"](g)
        f_optimizer = learning_args["f_optimizer_factory"](f)
        game_objective = learning_args["game_objective"]
        learner = SGDLearningDevF(
            game_objective=game_objective, g=g, f=f,
            g_optimizer=g_optimizer, f_optimizer=f_optimizer,
            dev_f_collection=dev_f_collection, e_dev_tilde=e_dev_tilde,
            final_g_optimizer_factory=self.default_g_opt_factory,
            video_plotter=video_plotter, do_averaging=False,
            max_num_epochs=6000, eval_freq=20, batch_size=1024,
            print_freq_mul=5, burn_in=1000, max_no_progress=20)
        learner.fit_from_tensors(x_train, y_train, z_train,
                                 x_dev, z_dev, y_dev, w_train=x_train,
                                 g_dev=g_dev, verbose=verbose)

    def predict(self, x_test):
        if self.g is None:
            raise AttributeError("Trying to call 'predict' before "
                                 "calling 'fit'")
        return self.g(x_test)
