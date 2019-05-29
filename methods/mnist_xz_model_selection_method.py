import torch
import torch.nn as nn
from torch.optim import Adam
from game_objectives.simple_moment_objective import OptimalMomentObjective
from learning.learning_dev_f import GradientDescentLearningDevF, SGDLearningDevF

from methods.abstract_method import AbstractMethod
from model_selection.f_history_model_selection_v2 import \
    FHistoryModelSelectionV2
from model_selection.f_history_model_selection_v3 import \
    FHistoryModelSelectionV3
from model_selection.learning_eval import FHistoryLearningEvalGradientDecent, \
    FHistoryLearningEvalSGD
from model_selection.learning_eval_nostop import FHistoryLearningEvalSGDNoStop
from model_selection.simple_model_eval import GradientDecentSimpleModelEval, \
    SGDSimpleModelEval
from models.cnn_models import LeakySoftmaxCNN, DefaultCNN
from models.mlp_model import MLPModel
from optimizers.oadam import OAdam
from optimizers.optimizer_factory import OptimizerFactory


class MNISTXZModelSelectionMethod(AbstractMethod):
    def __init__(self, enable_cuda=False):
        AbstractMethod.__init__(self)
        self.g = None
        self.f = None
        self.dev_f_collection = None

        g_models = [
            DefaultCNN(cuda=enable_cuda),
        ]
        f_models = [
            DefaultCNN(cuda=enable_cuda),
            # LeakySoftmaxCNN(input_c=1, input_h=28, input_w=28,
            #                 channel_sizes=[10, 20], kernel_sizes=[3, 3],
            #                 extra_padding=[0, 1], cuda=enable_cuda),
        ]

        g_learning_rates = [5e-6, 2e-6, 1e-6]
        # g_learning_rates = [0.00001]
        game_objective = OptimalMomentObjective()
        # g_learning_rates = [0.0005]
        # game_objectives = [OptimalMomentObjective(lambda_1=0.5)]
        learning_setups = []
        for g_lr in g_learning_rates:
            learning_setup = {
                "g_optimizer_factory": OptimizerFactory(
                    OAdam, lr=g_lr, betas=(0.5, 0.9)),
                "f_optimizer_factory": OptimizerFactory(
                    OAdam, lr=5.0*g_lr, betas=(0.5, 0.9)),
                "game_objective": game_objective
            }
            learning_setups.append(learning_setup)

        default_g_opt_factory = OptimizerFactory(
            Adam, lr=0.0001, betas=(0.5, 0.9))
        default_f_opt_factory = OptimizerFactory(
            Adam, lr=0.0001, betas=(0.5, 0.9))
        g_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=50, max_no_progress=10, batch_size=1024, eval_freq=1)
        f_simple_model_eval = SGDSimpleModelEval(
            max_num_epoch=50, max_no_progress=10, batch_size=1024, eval_freq=1)
        learning_eval = FHistoryLearningEvalSGDNoStop(
            num_epochs=60, eval_freq=1, batch_size=1024)
        self.model_selection = FHistoryModelSelectionV3(
            g_model_list=g_models,
            f_model_list=f_models,
            learning_args_list=learning_setups,
            default_g_optimizer_factory=default_g_opt_factory,
            default_f_optimizer_factory=default_f_opt_factory,
            g_simple_model_eval=g_simple_model_eval,
            f_simple_model_eval=f_simple_model_eval,
            learning_eval=learning_eval,
            psi_eval_burn_in=30, psi_eval_max_no_progress=10,
        )
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
            video_plotter=video_plotter, do_averaging=False, batch_size=1024,
            eval_freq=1, max_num_epochs=50, max_no_progress=10, burn_in=30,
            print_freq_mul=1)
        learner.fit_from_tensors(x_train, y_train, z_train,
                                 x_dev, z_dev, y_dev, w_train=x_train,
                                 g_dev=g_dev, verbose=verbose)

    def predict(self, x_test):
        if self.g is None:
            raise AttributeError("Trying to call 'predict' before "
                                 "calling 'fit'")
        self.g = self.g.eval()
        return self.g(x_test)


