import copy
import math
import random
import itertools
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas
from game_objectives.approximate_psi_objective import approx_psi_eval

from learning.abstract_learning import AbstractLearning
from plotting import PlotElement
from scenarios.abstract_scenario import AbstractScenario


class LearningTracker(object):
    def __init__(self, module, header):
        self._active = False
        self._module = module
        self._trigger = None
        self._log = dict()
        self._header = header

    def activate(self):
        self._active = True
        self._trigger = self._module.register_forward_hook(self.forward_hook)

    def deactivate(self):
        if self._active:
            self._active = False
            self._trigger.remove()

    def show_stats(self):
        if self._active:
            print(self._header)
            self.output_stats()
            self.gradient_stats()

    def gradient_stats(self):
        for name, kid in self._module.named_children():
            for n, p in kid.named_parameters():
                print("\tgrad norm(%s): %.2f" % (n, p.grad.norm(2).cpu().numpy()))

    def forward_hook(self, module, input, output):
        self._log["output_l2_norm"] = output.norm(2, dim=1).mean().cpu().numpy()
        self._log["output_min"] = output.min().cpu().numpy()
        self._log["output_max"] = output.max().cpu().numpy()
        self._log["output_mean"] = output.mean().cpu().numpy()

    def output_stats(self):
        print("\tl2-norm(output): %.2f" % self._log["output_l2_norm"])
        print("\tmin/max(output): %.2f/%.2f" % (self._log["output_min"], self._log["output_max"]))
        print("\tmean(output): %.2f" % self._log["output_mean"])


class AbstractLearningDevF(AbstractLearning):
    def __init__(self, game_objective, g, f, g_optimizer, f_optimizer,
                 dev_f_collection, e_dev_tilde, final_g_optimizer_factory,
                 burn_in, max_no_progress, video_plotter, eval_freq,
                 max_num_iterations, print_freq_mul, do_averaging):
        AbstractLearning.__init__(self, game_objective, g, f)
        self.max_num_iterations = max_num_iterations
        self.g_optimizer = g_optimizer
        self.f_optimizer = f_optimizer

        self.dev_f_collection = dev_f_collection
        self.e_dev_tilde = e_dev_tilde
        self.burn_in = burn_in
        self.max_no_progress = max_no_progress

        self.video_plotter = video_plotter
        self.print_freq_mul = print_freq_mul
        self.eval_freq = eval_freq

        self.final_g_optimizer_factory = final_g_optimizer_factory
        self.do_averaging = do_averaging

    def fit(self, scenario):
        assert isinstance(scenario, AbstractScenario)
        x_train, z_train, y_train, _, w_train = scenario.get_train_data()
        x_dev, z_dev, y_dev, g_dev, _ = scenario.get_dev_data()
        self.fit_from_tensors(x_train, y_train, z_train, x_dev, z_dev, y_dev,
                              g_dev=g_dev, w_train=w_train, verbose=True)

    def fit_from_tensors(self, x_train, y_train, z_train, x_dev, z_dev, y_dev,
                         g_dev=None, w_train=None, verbose=False):
        if w_train is not None:
            w_train_np = w_train.cpu().numpy()
        else:
            w_train_np = None
        y_train_cpu = y_train.cpu()
        y_dev_cpu = y_dev.cpu()
        if g_dev is not None:
            g_dev_cpu = g_dev.cpu()
        else:
            g_dev_cpu = None

        current_no_progress = 0
        eval_history = []
        g_state_history = []
        epsilon_dev_history = []
        epsilon_train_history = []

        g_of_x_train_list = []
        g_of_x_dev_list = []

        mse_list = []
        eval_list = []

        eval_freq = self.eval_freq
        print_freq = self.eval_freq * self.print_freq_mul

        for i in range(self.max_num_iterations):
            self.update_params_iter(i, x_train, z_train, y_train)

            # evaluate new g_of_x_dev against instruments
            if i % eval_freq == 0:
                self.f = self.f.eval()
                self.g = self.g.eval()

                g_of_x_train, f_of_z_train, obj_train = self.calc_f_g_obj(
                    x_train, z_train, y_train)
                g_of_x_dev, f_of_z_dev, obj_dev = self.calc_f_g_obj(
                    x_dev, z_dev, y_dev)

                if i >= self.burn_in and self.do_averaging:
                    g_of_x_train_list.append(g_of_x_train)
                    g_of_x_dev_list.append(g_of_x_dev)
                    epsilon_dev = (torch.stack(g_of_x_dev_list, 0).mean(0)
                                   - y_dev_cpu)
                    epsilon_train = (torch.stack(g_of_x_train_list, 0).mean(0)
                                     - y_train_cpu)
                else:
                    epsilon_dev = g_of_x_dev - y_dev_cpu
                    epsilon_train = g_of_x_train - y_train_cpu

                curr_eval = approx_psi_eval(epsilon_dev, self.dev_f_collection,
                                            self.e_dev_tilde)
                if g_dev_cpu is not None:
                    g_error = epsilon_dev + y_dev_cpu - g_dev_cpu
                    mse = float((g_error ** 2).mean())
                else:
                    mse = 0.0
                eval_list.append(curr_eval)
                mse_list.append(mse)
                if eval_history:
                    max_recent_eval = max(eval_history)
                else:
                    max_recent_eval = float("-inf")
                eval_history.append(curr_eval)
                epsilon_dev_history.append(epsilon_dev)
                epsilon_train_history.append(epsilon_train)
                g_state_history.append(copy.deepcopy(self.g.state_dict()))

                self.f = self.f.train()
                self.g = self.g.train()

            if self.video_plotter and i % print_freq == 0:
                frame = self.video_plotter.get_new_frame("iter = %d" % i)

                self.f = self.f.eval()
                self.g = self.g.eval()

                # plot f(z)
                frame.add_plot(PlotElement(
                    w_train_np, f_of_z_train.numpy(),
                    "estimated f(z)", normalize=True))

                # plot g(x)
                g_of_x_plot = epsilon_train_history[-1] + y_train_cpu
                frame.add_plot(PlotElement(w_train_np, g_of_x_plot.numpy(),
                                           "fitted g(x)"))

                self.f = self.f.train()
                self.g = self.g.train()

            if i % print_freq == 0 and verbose:
                mean_eval = np.mean(eval_history[-self.print_freq_mul:])
                print("iteration %d, dev-MSE=%f, train-loss=%f,"
                      " dev-loss=%f, mean-recent-eval=%f"
                      % (i, mse, obj_train, obj_dev, mean_eval))

            # check stopping conditions if we are past burn-in
            if i % self.eval_freq == 0 and i >= self.burn_in:
                if curr_eval > max_recent_eval:
                    current_no_progress = 0
                else:
                    current_no_progress += 1

                if current_no_progress >= self.max_no_progress:
                    break

        # plot relationship between MSE and eval
        if self.video_plotter:
            plt.figure()
            data = pandas.DataFrame({"eval": eval_list, "mse": mse_list})
            data.plot.scatter(x="eval", y="mse")
            plt.savefig("eval_mse.png")

        # finalize model
        max_i = max(range(len(eval_history)), key=lambda i_: eval_history[i_])
        if verbose:
            print("best iteration:", self.eval_freq * max_i)
        if self.do_averaging:
            g_final_train = epsilon_train_history[max_i] + y_train_cpu
            g_final_dev = epsilon_dev_history[max_i] + y_dev_cpu
            self.train_final_g(x_train, x_dev, g_final_train, g_final_dev)
        else:
            self.g.load_state_dict(g_state_history[max_i])

    def calc_f_g_obj(self, x, z, y, batch_size=1000):
        num_data = x.shape[0]
        num_batch = math.ceil(num_data * 1.0 / batch_size)
        g_of_x = None
        f_of_z = None
        obj_total = 0
        for b in range(num_batch):
            if b < num_batch - 1:
                batch_idx = list(range(b*batch_size, (b+1)*batch_size))
            else:
                batch_idx = list(range(b*batch_size, num_data))
            x_batch = x[batch_idx]
            z_batch = z[batch_idx]
            y_batch = y[batch_idx]
            g_obj, _ = self.game_objective.calc_objective(
                self.g, self.f, x_batch, z_batch, y_batch)
            g_of_x_batch = self.g(x_batch).detach().cpu()
            f_of_z_batch = self.f(z_batch).detach().cpu()
            if b == 0:
                g_of_x = g_of_x_batch
                f_of_z = f_of_z_batch
            else:
                g_of_x = torch.cat([g_of_x, g_of_x_batch], dim=0)
                f_of_z = torch.cat([f_of_z, f_of_z_batch], dim=0)
            obj_total += float(g_obj.detach().cpu()) * len(batch_idx) * 1.0 / num_data
        return g_of_x, f_of_z, float(g_obj.detach().cpu())

    def train_final_g(self, x_train, x_dev, g_final_train, g_final_dev):
        loss_history = []
        g_state_history = []
        current_no_progress = 0
        optimizer = self.final_g_optimizer_factory(self.g)

        for i in range(self.max_num_iterations):

            self.update_final_params_iter(optimizer, x_train, g_final_train)

            if i % self.eval_freq == 0:
                self.f = self.f.eval()
                self.g = self.g.eval()

                # calculate dev loss
                dev_loss = float(((self.g(x_dev) - g_final_dev) ** 2).mean())
                if loss_history:
                    min_loss = min(loss_history)
                else:
                    min_loss = float("-inf")
                loss_history.append(dev_loss)
                g_state_history.append(copy.deepcopy(self.g.state_dict()))

                if dev_loss >= min_loss:
                    current_no_progress += 1
                else:
                    current_no_progress = 0

                self.f = self.f.train()
                self.g = self.g.train()

                # break if necessary
                if current_no_progress >= self.max_no_progress:
                    break

        min_i = min(range(len(loss_history)), key=lambda i_: loss_history[i_])
        self.g.load_state_dict(g_state_history[min_i])

    def update_params_iter(self, iteration, x, z, y):
        raise NotImplementedError()

    def update_final_params_iter(self, optimizer, x_train, g_final_train):
        raise NotImplementedError()


class GradientDescentLearningDevF(AbstractLearningDevF):
    def __init__(self, game_objective, g, f, g_optimizer, f_optimizer,
                 dev_f_collection, e_dev_tilde, final_g_optimizer_factory,
                 burn_in=2000, max_no_progress=20, video_plotter=None,
                 eval_freq=50, max_num_iterations=8000,
                 print_freq_mul=4, show_debug_info=False, do_averaging=True):
        AbstractLearningDevF.__init__(
            self, game_objective=game_objective, g=g, f=f,
            g_optimizer=g_optimizer, f_optimizer=f_optimizer,
            dev_f_collection=dev_f_collection, e_dev_tilde=e_dev_tilde,
            final_g_optimizer_factory=final_g_optimizer_factory,
            burn_in=burn_in, max_no_progress=max_no_progress,
            eval_freq=eval_freq, max_num_iterations=max_num_iterations,
            print_freq_mul=print_freq_mul, video_plotter=video_plotter,
            do_averaging=do_averaging)
        self._show_debug_info = show_debug_info
        self.g_tracker = LearningTracker(self.g, "g")
        self.f_tracker = LearningTracker(self.f, "f")

    def update_params_iter(self, iteration, x, z, y):
        print_freq = self.eval_freq * self.print_freq_mul
        if self._show_debug_info and iteration % print_freq == 0:
            self.g_tracker.activate()
            self.f_tracker.activate()

        g_obj, f_obj = self.game_objective.calc_objective(
            self.g, self.f, x, z, y)

        # do single step optimization on f and g
        self.g_optimizer.zero_grad()
        g_obj.backward(retain_graph=True)

        self.g_tracker.show_stats()
        self.g_optimizer.step()

        self.f_optimizer.zero_grad()
        f_obj.backward()

        self.f_tracker.show_stats()
        self.f_optimizer.step()

        self.g_tracker.deactivate()
        self.f_tracker.deactivate()

    def update_final_params_iter(self, optimizer, x_train, g_final_train):
        optimizer.zero_grad()
        loss = ((self.g(x_train) - g_final_train) ** 2).mean()
        loss.backward()
        optimizer.step()


class SGDLearningDevF(AbstractLearningDevF):
    def __init__(self, game_objective, g, f, g_optimizer, f_optimizer,
                 dev_f_collection, e_dev_tilde, final_g_optimizer_factory,
                 burn_in=200, max_no_progress=10, video_plotter=None,
                 eval_freq=10, max_num_epochs=2000, batch_size=256,
                 print_freq_mul=2, do_averaging=True):
        AbstractLearningDevF.__init__(
            self, game_objective=game_objective, g=g, f=f,
            g_optimizer=g_optimizer, f_optimizer=f_optimizer,
            dev_f_collection=dev_f_collection, e_dev_tilde=e_dev_tilde,
            final_g_optimizer_factory=final_g_optimizer_factory,
            burn_in=burn_in, max_no_progress=max_no_progress,
            eval_freq=eval_freq, max_num_iterations=max_num_epochs,
            print_freq_mul=print_freq_mul, video_plotter=video_plotter,
            do_averaging=do_averaging)
        self.batch_size = batch_size

    def update_params_iter(self, iteration, x, z, y):
        num_data = x.shape[0]
        num_batch = math.ceil(num_data * 1.0 / self.batch_size)
        train_idx = list(range(num_data))
        random.shuffle(train_idx)
        idx_iter = itertools.cycle(train_idx)

        # loop through training data
        for _ in range(num_batch):
            batch_idx = [next(idx_iter) for _ in range(self.batch_size)]
            x_batch = x[batch_idx]
            z_batch = z[batch_idx]
            y_batch = y[batch_idx]
            g_obj, f_obj = self.game_objective.calc_objective(
                self.g, self.f, x_batch, z_batch, y_batch)

            # do single step optimization on f and g
            self.g_optimizer.zero_grad()
            g_obj.backward(retain_graph=True)
            self.g_optimizer.step()

            self.f_optimizer.zero_grad()
            f_obj.backward()
            self.f_optimizer.step()

    def update_final_params_iter(self, optimizer, x_train, g_final_train):
        num_data = x_train.shape[0]
        num_batch = math.ceil(num_data * 1.0 / self.batch_size)
        train_idx = list(range(num_data))
        random.shuffle(train_idx)
        idx_iter = itertools.cycle(train_idx)

        # loop through training data
        for _ in range(num_batch):
            batch_idx = [next(idx_iter) for _ in range(self.batch_size)]
            x_batch = x_train[batch_idx]
            g_batch = g_final_train[batch_idx]
            optimizer.zero_grad()
            loss = ((self.g(x_batch) - g_batch) ** 2).mean()
            loss.backward()
            optimizer.step()

