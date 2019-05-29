from collections import deque
import math
import random
import itertools
import torch
from model_selection.abstract_learning_eval import AbstractLearningEval


class FHistoryLearningEvalNoStop(AbstractLearningEval):
    def __init__(self, num_iter=8000, history_len=100,
                 eval_freq=20, print_freq=100):
        AbstractLearningEval.__init__(self)
        self.num_iter = num_iter
        self.history_len = history_len
        self.eval_freq = eval_freq
        self.print_freq = print_freq

    def eval(self, x_train, z_train, y_train, x_dev, z_dev, y_dev,
             g, f, g_optimizer, f_optimizer, game_objective):
        AbstractLearningEval.__init__(self)
        epsilon_dev_history = []
        f_of_z_dev_history = []
        y_dev_cpu = y_dev.cpu()

        for i in range(self.num_iter):
            self.do_training_update(x_train, z_train, y_train, g, f,
                                    g_optimizer, f_optimizer, game_objective)

            if i % self.eval_freq == 0:

                f = f.eval()
                g = g.eval()

                # calculate f and g on dev, and update histories
                epsilon_dev = self.calc_function_batched(g, x_dev) - y_dev_cpu
                epsilon_dev_history.append(epsilon_dev)

                f_of_z_dev = self.calc_function_batched(f, z_dev)
                f_of_z_dev_history.append(f_of_z_dev)

                f = f.train()
                g = g.train()

        return epsilon_dev_history, f_of_z_dev_history

    def calc_function_batched(self, function, data, batch_size=1000):
        num_data = data.shape[0]
        num_batch = math.ceil(num_data * 1.0 / batch_size)
        out = None
        for b in range(num_batch):
            if b < num_batch - 1:
                batch_idx = list(range(b*batch_size, (b+1)*batch_size))
            else:
                batch_idx = list(range(b*batch_size, num_data))
            data_batch = data[batch_idx]
            out_batch = function(data_batch).detach().cpu()
            if b == 0:
                out = out_batch
            else:
                out = torch.cat([out, out_batch], dim=0)
        return out

    def do_training_update(self, x_train, z_train, y_train, g, f,
                           g_optimizer, f_optimizer, game_objective):
        raise NotImplementedError()


class FHistoryLearningEvalSGDNoStop(FHistoryLearningEvalNoStop):
    def __init__(self, num_epochs=50, batch_size=256, history_len=100,
                 eval_freq=10, print_freq=50):
        FHistoryLearningEvalNoStop.__init__(
            self, num_iter=num_epochs, history_len=history_len,
            eval_freq=eval_freq, print_freq=print_freq)
        AbstractLearningEval.__init__(self)
        self.batch_size = batch_size

    def do_training_update(self, x_train, z_train, y_train, g, f,
                           g_optimizer, f_optimizer, game_objective):
        num_train = x_train.shape[0]
        num_batch = math.ceil(num_train * 1.0 / self.batch_size)
        train_idx = list(range(num_train))
        random.shuffle(train_idx)
        train_idx_iter = itertools.cycle(train_idx)
        # loop through training data
        for _ in range(num_batch):
            batch_idx = [next(train_idx_iter) for _ in range(self.batch_size)]
            x_batch = x_train[batch_idx]
            z_batch = z_train[batch_idx]
            y_batch = y_train[batch_idx]
            g_obj, f_obj = game_objective.calc_objective(
                g, f, x_batch, z_batch, y_batch)

            # do single step optimization on f and g
            g_optimizer.zero_grad()
            g_obj.backward(retain_graph=True)
            g_optimizer.step()

            f_optimizer.zero_grad()
            f_obj.backward()
            f_optimizer.step()


class FHistoryLearningEvalGradientDecentNoStop(FHistoryLearningEvalNoStop):
    def __init__(self, num_iter=6000, history_len=100,
                 eval_freq=50, print_freq=100):
        FHistoryLearningEvalNoStop.__init__(
            self, num_iter=num_iter, history_len=history_len,
            eval_freq=eval_freq, print_freq=print_freq)
        AbstractLearningEval.__init__(self)

    def do_training_update(self, x_train, z_train, y_train, g, f,
                           g_optimizer, f_optimizer, game_objective):
        g_obj, f_obj = game_objective.calc_objective(
            g, f, x_train, z_train, y_train)

        # do single step optimization on f and g
        g_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)
        g_optimizer.step()

        f_optimizer.zero_grad()
        f_obj.backward()
        f_optimizer.step()
