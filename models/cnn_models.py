import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNModel(nn.Module):
    def __init__(self, input_c, input_h, input_w, kernel_sizes, extra_padding,
                 activation=None):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0

        cnn_layers = []
        c, h, w = input_c, input_h, input_w
        for k, ep in zip(kernel_sizes, extra_padding):
            p = (k - 1) // 2 + ep
            new_layers = [nn.Conv2d(c, 2 * c, kernel_size=k, padding=p),
                          nn.MaxPool2d(2), activation()]
            cnn_layers.extend(new_layers)
            h = h // 2 + ep
            w = w // 2 + ep
            c *= 2
        self.cnn = nn.Sequential(*cnn_layers)
        self.linear = nn.Linear(h * w * c, 1)
        self.linear_input_dim = h * w * c

    def initialize(self):
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        nn.init.xavier_normal_(self.linear.weight.data, gain=1.0)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, data):
        # print(data.shape)
        data = self.cnn(data)
        return self.linear(data.view(-1, self.linear_input_dim))


class SimpleCNNModelV2(nn.Module):
    def __init__(self, input_c, input_h, input_w, kernel_sizes, extra_padding,
                 hidden_c_sizes, final_c=None, activation=None):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0

        cnn_layers = []
        h, w = input_h, input_w
        for output_c, k, ep in zip(hidden_c_sizes, kernel_sizes, extra_padding):
            p = (k - 1) // 2 + ep
            new_layers = [nn.Conv2d(input_c, output_c, kernel_size=k,
                                    padding=p),
                          nn.MaxPool2d(2), activation()]
            cnn_layers.extend(new_layers)
            h = h // 2 + ep
            w = w // 2 + ep
            input_c = output_c
        if final_c:
            final_cnn = nn.Conv2d(hidden_c_sizes[-1], final_c, kernel_size=1)
            cnn_layers.append(final_cnn)
        else:
            final_c = hidden_c_sizes[-1]

        self.cnn = nn.Sequential(*cnn_layers)
        self.final_pool = nn.MaxPool2d(kernel_size=(h, w))
        self.linear = nn.Linear(final_c, 1)
        self.linear_input_dim = final_c

    def initialize(self):
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        nn.init.xavier_normal_(self.linear.weight.data, gain=1.0)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, data):
        # print(data.shape)
        data = self.cnn(data)
        data = self.final_pool(data).view(-1, self.linear_input_dim)
        out = self.linear(data)
        print(data[:5], out[:5])
        return out


class SimpleCNNModelV3(nn.Module):
    def __init__(self, input_c, input_h, input_w, kernel_sizes, extra_padding,
                 activation=None):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0

        cnn_layers = []
        c, h, w = input_c, input_h, input_w
        for k, ep in zip(kernel_sizes, extra_padding):
            p = (k - 1) // 2 + ep
            new_layers = [nn.Conv2d(c, 2 * c, kernel_size=k, padding=p),
                          nn.MaxPool2d(2), activation()]
            cnn_layers.extend(new_layers)
            h = h // 2 + ep
            w = w // 2 + ep
            c *= 2
        self.cnn = nn.Sequential(*cnn_layers)
        self.final_pool = nn.AvgPool2d(kernel_size=(h, w))
        self.linear_1 = nn.Linear(h * w * c, 20)
        self.linear_input_dim = h * w * c
        self.linear_2 = nn.Linear(20, 3)
        self.linear_3 = nn.Linear(3, 1)

    def initialize(self):
        for layer in self.cnn:
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        leaky_relu_gain = nn.init.calculate_gain("leaky_relu")
        nn.init.xavier_normal_(self.linear_1.weight.data, gain=leaky_relu_gain)
        nn.init.zeros_(self.linear_1.bias.data)
        nn.init.xavier_normal_(self.linear_2.weight.data, gain=leaky_relu_gain)
        nn.init.zeros_(self.linear_2.bias.data)
        nn.init.xavier_normal_(self.linear_3.weight.data, gain=1.0)
        nn.init.zeros_(self.linear_3.bias.data)

    def forward(self, data):
        # print(data.shape)
        data = self.cnn(data)
        data = F.leaky_relu(self.linear_1(data.view(-1, self.linear_input_dim)))
        data = F.leaky_relu(self.linear_2(data))
        return self.linear_3(data)


class LeakySoftmaxCNN(nn.Module):
    def __init__(self, input_c, input_h, input_w, kernel_sizes, extra_padding,
                 channel_sizes, final_c=5, activation=None, cuda=False):
        nn.Module.__init__(self)
        if activation is None:
            activation = nn.ReLU
        if activation.__class__.__name__ == "LeakyReLU":
            self.gain = nn.init.calculate_gain("leaky_relu",
                                               activation.negative_slope)
        else:
            activation_name = activation.__class__.__name__.lower()
            try:
                self.gain = nn.init.calculate_gain(activation_name)
            except ValueError:
                self.gain = 1.0
        self.input_c = input_c
        self.input_h = input_h
        self.input_w = input_w
        self.kernel_sizes = kernel_sizes
        self.extra_padding = extra_padding
        self.channel_sizes = channel_sizes
        self.final_c = final_c
        self.activation = activation
        self.use_cuda = cuda
        self.initialize()

    def initialize(self):
        cnn_layers = []
        c, h, w = self.input_c, self.input_h, self.input_w
        for i, (k, ep) in enumerate(zip(self.kernel_sizes, self.extra_padding)):
            c_in = self.channel_sizes[i-1] if i > 0 else c
            c_out = self.channel_sizes[i]
            p = (k - 1) // 2 + ep
            new_layers = [nn.Conv2d(c_in, c_out, kernel_size=k, padding=p),
                          self.activation(),
                          nn.MaxPool2d(2),
                          nn.Dropout(p=0.0)]
            cnn_layers.extend(new_layers)
            h = h // 2 + ep
            w = w // 2 + ep
        # cnn_layers.append(nn.Conv2d(channel_sizes[-1], final_c,
        #                             kernel_size=1, padding=0))
        self.cnn = nn.Sequential(*cnn_layers)
        self.linear_1 = nn.Linear(h * w * self.channel_sizes[-1], 200)
        self.linear_input_dim = h * w * self.channel_sizes[-1]
        # self.linear_1 = nn.Linear(h * w * final_c, 200)
        # self.linear_input_dim = h * w * final_c
        self.linear_2 = nn.Linear(200, 10)
        self.linear_3 = nn.Linear(10, 1)
        self.bn = nn.BatchNorm1d(10)
        self.double()
        if self.use_cuda:
            self.cuda()

    def forward(self, data):
        # print(data.shape)
        data = F.dropout(self.cnn(data), p=0.0, training=self.training)
        # data = self.cnn(data)
        data = F.dropout(
            F.leaky_relu(self.linear_1(data.view(-1, self.linear_input_dim))),
            p=0.0, training=self.training)

        leaky_class_weights = F.leaky_relu(self.linear_2(data))
        class_probs = F.softmax(self.linear_2(data), dim=1)
        total_weight = leaky_class_weights.sum(1).view(-1, 1)
        data = leaky_class_weights * 0.01 + total_weight * class_probs * 0.99
        data = self.bn(data)
        return self.linear_3(data)


class DefaultCNN(LeakySoftmaxCNN):
    def __init__(self, cuda):
        LeakySoftmaxCNN.__init__(
            self, input_c=1, input_h=28, input_w=28, channel_sizes=[20, 50],
            kernel_sizes=[3, 3], extra_padding=[0, 1],
            activation=nn.LeakyReLU, cuda=cuda)

    def forward(self, data):
        data = data.view(data.shape[0], 1, 28, 28)
        return LeakySoftmaxCNN.forward(self, data)
