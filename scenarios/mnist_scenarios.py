from collections import defaultdict
import random
import numpy as np
import torch
from torchvision import datasets, transforms

from scenarios.abstract_scenario import AbstractScenario
from scenarios.toy_scenarios import AGMMZoo


class AbstractMNISTScenario(AbstractScenario):
    def __init__(self, use_x_images, use_z_images, g_function):
        AbstractScenario.__init__(self)
        # mnist_train = datasets.MNIST(
        #     "datasets", train=True, download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        # )
        # mnist_test = datasets.MNIST(
        #     "datasets", train=False, download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        # )
        # train_images = mnist_train.train_data.unsqueeze(1).numpy()
        # test_images = mnist_test.test_data.unsqueeze(1).numpy()
        # train_labels = mnist_train.train_labels.numpy()
        # test_labels = mnist_test.test_labels.numpy()
        # self.images = np.concatenate([train_images, test_images], axis=0)
        # self.labels = np.concatenate([train_labels, test_labels], axis=0)
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST("datasets", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=60000)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST("datasets", train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])), batch_size=10000)
        train_data, test_data = list(train_loader), list(test_loader)
        images_list = [train_data[0][0].numpy(), test_data[0][0].numpy()]
        labels_list = [train_data[0][1].numpy(), test_data[0][1].numpy()]
        self.images = np.concatenate(images_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)
        idx = list(range(self.images.shape[0]))
        random.shuffle(idx)
        self.images = self.images[idx]
        self.labels = self.labels[idx]
        self.data_i = 0

        self.toy_scenario = AGMMZoo(
            g_function=g_function, two_gps=False, n_instruments=1,
            iv_strength=0.5)

        self.use_x_images = use_x_images
        self.use_z_images = use_z_images

    def _sample_images(self, sample_digits, images, labels):
        # image_array = np.zeros(shape=(len(sample_digits), 1, 28, 28))
        # for d in range(10):
        #     d_idx = np.array([int(d_) for d_ in labels if d_ == d])
        #     fill_idx = np.array([int(d_) for d_ in sample_digits if d_ == d])
        #     num_digits = len(d_idx)
        #     num_fill = len(fill_idx)
        #     d_images = images[d_idx]
        #     idx_sample = np.random.choice(list(range(num_digits)), num_fill)
        #     image_sample = d_images[idx_sample]
        #     image_array[fill_idx] = image_sample
        # return image_array
        digit_dict = defaultdict(list)
        for l, image in zip(labels, images):
            digit_dict[int(l)].append(image)
        images = np.stack([random.choice(digit_dict[int(d)])
                           for d in sample_digits], axis=0)
        return images

    @staticmethod
    def _g_step_function(x):
        return 5 * (x >= 5)

    def generate_data(self, num_data, **kwargs):
        idx = list(range(self.data_i, self.data_i+num_data))
        images = self.images[idx]
        labels = self.labels[idx]
        self.data_i += num_data
        # images = self.images
        # labels = self.labels

        # z_float = np.random.uniform(-0.5, 9.5, size=num_data)
        # zeta = np.random.normal(0, 2.0, size=num_data)
        # # ZETA_DICT = {-5: 1, -4: 2, -3: 3, -2: 4, -1: 5, 0: 6,
        # #              1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        # # ZETA_VALS = np.array(list(ZETA_DICT.keys()))
        # # ZETA_PROBS = np.array(list(ZETA_DICT.values())) ** 2
        # # ZETA_PROBS = ZETA_PROBS / ZETA_PROBS.sum()
        # # zeta = np.random.choice(ZETA_VALS, p=ZETA_PROBS, size=num_data)
        # # z_float = np.random.randint(0, 10, size=num_data)
        #
        # x_float = z_float + zeta
        #
        # if self.use_x_images:
        #     x_digits = np.clip(x_float, 0, 9).round()
        #     x = self._sample_images(x_digits, images, labels).reshape(-1, 1, 28, 28)
        #     g = self._g_step_function(x_digits).reshape(-1, 1)
        #     w = x_digits.reshape(-1, 1)
        # else:
        #     x = x_float.reshape(-1, 1)
        #     g = self._g_step_function(x)
        #     w = x_float.reshape(-1, 1)
        #
        # if self.use_z_images:
        #     z_digits = np.clip(z_float, 0, 9).round()
        #     z = self._sample_images(z_digits, images, labels).reshape(-1, 1, 28, 28)
        # else:
        #     z = z_float.reshape(-1, 1)
        #
        # epsilon = np.random.normal(size=(num_data,1))
        # y = g + 2 * zeta.reshape(-1, 1) + 0.5 * epsilon.reshape(-1, 1)
        #
        # return x, z, y, g, w

        toy_x, toy_z, toy_y, toy_g, _ = self.toy_scenario.generate_data(num_data)
        if self.use_x_images:
            x_digits = np.clip(1.5*toy_x[:, 0] + 5.0, 0, 9).round()
            x = self._sample_images(x_digits, images, labels).reshape(-1, 1, 28, 28)
            # g = self._g_step_function(x_digits).reshape(-1, 1)
            g = self.toy_scenario._true_g_function_np((x_digits - 5.0) / 1.5).reshape(-1, 1)
            w = x_digits.reshape(-1, 1)
        else:
            x = toy_x.reshape(-1, 1) * 1.5 + 5.0
            g = toy_g.reshape(-1, 1)
            w = toy_x.reshape(-1, 1) * 1.5 + 5.0

        if self.use_z_images:
            z_digits = np.clip(1.5*toy_z[:, 0] + 5.0, 0, 9).round()
            z = self._sample_images(z_digits, images, labels).reshape(-1, 1, 28, 28)
        else:
            z = toy_z.reshape(-1, 1)

        # print(np.stack([w[:20, 0], g[:20, 0]]))
        return x, z, toy_y, g, w

    def true_g_function(self, x):
        raise NotImplementedError()


class MNISTScenarioX(AbstractMNISTScenario):
    def __init__(self, g_function="abs"):
        AbstractMNISTScenario.__init__(
            self, use_x_images=True, use_z_images=False, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()


class MNISTScenarioZ(AbstractMNISTScenario):
    def __init__(self, g_function="abs"):
        AbstractMNISTScenario.__init__(
            self, use_x_images=False, use_z_images=True, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()


class MNISTScenarioXZ(AbstractMNISTScenario):
    def __init__(self, g_function="abs"):
        AbstractMNISTScenario.__init__(
            self, use_x_images=True, use_z_images=True, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()

class MNISTScenarioNone(AbstractMNISTScenario):
    def __init__(self, g_function="abs"):
        AbstractMNISTScenario.__init__(
            self, use_x_images=False, use_z_images=False, g_function=g_function)

    def true_g_function(self, x):
        raise NotImplementedError()



class AbstractMNISTScenarioOld(AbstractScenario):
    def __init__(self):
        AbstractScenario.__init__(self)
        mnist_train = datasets.MNIST(
            "datasets", train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        self.images = mnist_train.train_data.unsqueeze(1).numpy()
        self.labels = mnist_train.train_labels.numpy()
        # mnist_train = datasets.MNIST(
        #     "datasets", train=True, download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        # )
        # mnist_test = datasets.MNIST(
        #     "datasets", train=False, download=True,
        #     transform=transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,))
        #     ])
        # )
        # train_images = mnist_train.train_data.unsqueeze(1).numpy()
        # test_images = mnist_test.test_data.unsqueeze(1).numpy()
        # train_labels = mnist_train.train_labels.numpy()
        # test_labels = mnist_test.test_labels.numpy()
        # self.images = np.concatenate([train_images, test_images], axis=0)
        # self.labels = np.concatenate([train_labels, test_labels], axis=0)

        self.data_i = 0
        self.idx = list(range(self.images.shape[0]))
        random.shuffle(self.idx)

    def generate_data(self, num_data, **kwargs):
        idx = self.idx[self.data_i:self.data_i+num_data]
        images = self.images[idx]
        labels = self.labels[idx]
        self.data_i += num_data
        return self.generate_mnist_data(images, labels, num_data)

    def generate_mnist_data(self, images, labels, num_data):
        raise NotImplementedError()

    def true_g_function(self, x):
        raise NotImplementedError()


class MNISTScenarioXZOld(AbstractMNISTScenarioOld):
    def __init__(self, clamp=True):
        AbstractMNISTScenarioOld.__init__(self)
        self.digit_dict = defaultdict(list)
        for l, image in zip(self.labels, self.images):
            self.digit_dict[int(l)].append(image)
        self.clamp = clamp

    def generate_mnist_data(self, images, labels, num_data):
        epsilon = np.random.normal(size=num_data)
        # zeta = np.random.randint(0, 2, size=num_data) * 2 - 1
        zeta_dict = {-5: 1, -4: 2, -3: 3, -2: 4, -1: 5, 0: 6,
                     1: 5, 2: 4, 3: 3, 4: 2, 5: 1}
        zeta_vals = np.array(list(zeta_dict.keys()))
        zeta_probs = np.array(list(zeta_dict.values())) ** 2
        zeta_probs = zeta_probs / zeta_probs.sum()
        zeta = np.random.choice(zeta_vals, p=zeta_probs, size=num_data)
        w = labels + zeta
        if self.clamp:
            w = np.clip(w, 0, 9)
        else:
            w %= 10
        digit_dict = defaultdict(list)
        for l, image in zip(labels, images):
            digit_dict[int(l)].append(image)
        x = np.stack([random.choice(digit_dict[int(l)]) for l in w],
                     axis=0)
        g = self.true_g_function(w)
        y = g + 2 * zeta + 0.5 * epsilon
        z = images
        # idx = list(range(num_data))
        # random.shuffle(idx)
        # z = z[idx]
        return x, z, y.reshape(-1, 1), g.reshape(-1, 1), w.reshape(-1, 1)

    def true_g_function(self, w):
        return 5.0 * (w >= 5)
        # return np.maximum(3 * w - 11.0, 0.25 * w)
