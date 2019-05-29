from collections import deque
import math
import random
import itertools
import torch
from model_selection.abstract_learning_eval import AbstractLearningEval


def f_history_g_eval(epsilon_dev, f_of_z_dev_history):
    eval_list = []
    for f_of_z in f_of_z_dev_history:
        raw_moment = f_of_z.mul(epsilon_dev).mean()
        denominator = (f_of_z ** 2).mul(epsilon_dev ** 2).mean() ** 0.5
        eval_list.append(-1.0 * float(raw_moment) / float(denominator))
    return min(eval_list)


class FHistoryLearningEval(AbstractLearningEval):
    def __init__(self, max_num_iter=8000, burn_in=2000, history_len=100,
                 eval_freq=20, print_freq=100, max_no_progress=10,
                 do_averaging=True):
        AbstractLearningEval.__init__(self)
        self.max_num_iter = max_num_iter
        self.burn_in = burn_in
        self.history_len = history_len
        self.eval_freq = eval_freq
        self.print_freq = print_freq
        self.max_no_progress = max_no_progress
        self.do_averaging = do_averaging

    def eval(self, x_train, z_train, y_train, x_dev, z_dev, y_dev,
             g, f, g_optimizer, f_optimizer, game_objective):
        AbstractLearningEval.__init__(self)
        current_no_progress = 0
        f_of_z_dev_history = []
        history_len = max(math.ceil(self.burn_in / self.eval_freq),
                          self.history_len)
        assert self.max_no_progress < history_len
        epsilon_dev_history = deque(maxlen=history_len)
        eval_history = deque(maxlen=history_len)
        g_of_x_dev_list = []
        y_dev_cpu = y_dev.cpu()

        for i in range(self.max_num_iter):
            self.do_training_update(x_train, z_train, y_train, g, f,
                                    g_optimizer, f_optimizer, game_objective)

            if i % self.eval_freq == 0:

                f = f.eval()
                g = g.eval()

                # calculate f and g on dev, and update histories
                g_of_x_dev = self.calc_function_batched(g, x_dev)
                if i >= self.burn_in and self.do_averaging:
                    g_of_x_dev_list.append(g_of_x_dev)
                    mean_g_of_x_dev = torch.stack(g_of_x_dev_list, 0).mean(0)
                    epsilon_dev = mean_g_of_x_dev - y_dev_cpu
                else:
                    epsilon_dev = g_of_x_dev - y_dev_cpu
                f_of_z_dev = self.calc_function_batched(f, z_dev)
                epsilon_dev_history.append(epsilon_dev)

                # update recent eval history with new f_of_z
                for j, old_eval in enumerate(eval_history):
                    old_epsilon = epsilon_dev_history[j]
                    learning_eval = f_history_g_eval(old_epsilon, [f_of_z_dev])
                    eval_history[j] = min(learning_eval, old_eval)

                # evaluate new g_of_x_dev
                if eval_history:
                    max_recent_eval = max(eval_history)
                else:
                    max_recent_eval = float("-inf")
                f_of_z_dev_history.append(f_of_z_dev)
                epsilon_dev_history.append(epsilon_dev)
                curr_eval = f_history_g_eval(epsilon_dev, f_of_z_dev_history)
                eval_history.append(curr_eval)

                f = f.train()
                g = g.train()

                # check stopping conditions if we are past burn-in
                if i >= self.burn_in:
                    if curr_eval > max_recent_eval:
                        current_no_progress = 0
                    else:
                        current_no_progress += 1

                    if current_no_progress >= self.max_no_progress:
                        break

        max_i = max(range(len(eval_history)), key=lambda i_: eval_history[i_])
        return eval_history[max_i], epsilon_dev_history[max_i], f_of_z_dev_history

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


class FHistoryLearningEvalSGD(FHistoryLearningEval):
    def __init__(self, max_num_epochs=2000, batch_size=256, burn_in=200,
                 history_len=100, max_no_progress=10, eval_freq=10,
                 print_freq=50, do_averaging=False):
        FHistoryLearningEval.__init__(
            self, max_num_iter=max_num_epochs, burn_in=burn_in,
            history_len=history_len, eval_freq=eval_freq,
            print_freq=print_freq, max_no_progress=max_no_progress,
            do_averaging=do_averaging)
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


class FHistoryLearningEvalGradientDecent(FHistoryLearningEval):
    def __init__(self, max_num_iter=8000, burn_in=2000, history_len=100,
                 eval_freq=20, print_freq=100, max_no_progress=10,
                 do_averaging=True):
        FHistoryLearningEval.__init__(
            self, max_num_iter=max_num_iter, burn_in=burn_in,
            history_len=history_len, eval_freq=eval_freq,
            print_freq=print_freq, max_no_progress=max_no_progress,
            do_averaging=do_averaging)
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
