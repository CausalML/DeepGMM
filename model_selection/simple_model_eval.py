import math
import random
import itertools

import numpy as np
import torch


class AbstractSimpleModelEval(object):
    def __init__(self):
        pass

    def eval(self, f, f_optimizer, x_train, y_train, x_dev, y_dev):
        raise NotImplementedError()


class StandardSimpleModelEval(AbstractSimpleModelEval):
    def __init__(self, max_no_progress, eval_freq, max_num_iter):
        AbstractSimpleModelEval.__init__(self)
        self.max_no_progress = max_no_progress
        self.max_num_iter = max_num_iter
        self.eval_freq = eval_freq

    def eval(self, f, f_optimizer, x_train, y_train, x_dev, y_dev):
        min_loss = float("inf")
        min_loss_std = float("inf")
        current_no_progress = 0
        num_dev = x_dev.shape[0]
        y_dev_cpu = y_dev.cpu()

        for i in range(self.max_num_iter):

            self.do_training_update(f, f_optimizer, x_train, y_train)

            if i % self.eval_freq == 0:
                f = f.eval()
                # calculate dev loss
                f_x_dev = self.calc_function_batched(f, x_dev)
                loss_vec = ((f_x_dev - y_dev_cpu) ** 2).numpy()
                dev_loss = float(np.mean(loss_vec))
                dev_loss_std = float(np.std(loss_vec) / np.sqrt(num_dev))
                if dev_loss >= min_loss:
                    current_no_progress += 1
                else:
                    current_no_progress = 0
                    min_loss = dev_loss
                    min_loss_std = dev_loss_std

                f = f.train()
                # break if necessary
                if current_no_progress >= self.max_no_progress:
                    break

        return -min_loss, min_loss_std

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

    def do_training_update(self, f, f_optimizer, x_train, y_train):
        raise NotImplementedError()


class SGDSimpleModelEval(StandardSimpleModelEval):
    def __init__(self, batch_size=256, max_no_progress=10, max_num_epoch=1000,
                 eval_freq=10):
        StandardSimpleModelEval.__init__(self, max_no_progress=max_no_progress,
                                         max_num_iter=max_num_epoch,
                                         eval_freq=eval_freq)
        self.batch_size = batch_size

    def do_training_update(self, f, f_optimizer, x_train, y_train):
        num_train = x_train.shape[0]
        num_batch = math.ceil(num_train * 1.0 / self.batch_size)
        train_idx = list(range(num_train))
        random.shuffle(train_idx)
        train_idx_iter = itertools.cycle(train_idx)

        # loop through training data
        for _ in range(num_batch):
            batch_idx = [next(train_idx_iter)
                         for _ in range(self.batch_size)]
            x_batch = x_train[batch_idx]
            y_batch = y_train[batch_idx]
            f_optimizer.zero_grad()
            loss = ((f(x_batch) - y_batch) ** 2).mean()
            loss.backward()
            f_optimizer.step()


class GradientDecentSimpleModelEval(StandardSimpleModelEval):
    def __init__(self, max_no_progress=10, max_num_iter=8000, eval_freq=20):
        StandardSimpleModelEval.__init__(self, max_no_progress=max_no_progress,
                                         max_num_iter=max_num_iter,
                                         eval_freq=eval_freq)

    def do_training_update(self, f, f_optimizer, x_train, y_train):
        f_optimizer.zero_grad()
        loss = ((f(x_train) - y_train) ** 2).mean()
        loss.backward()
        f_optimizer.step()
