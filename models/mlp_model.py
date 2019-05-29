import torch
import torch.nn as nn


class MLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, activation=None,
                 last_layer=None, num_out=1):
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

        if len(layer_widths) == 0:
            layers = [nn.Linear(input_dim, num_out)]
        else:
            num_layers = len(layer_widths)
            layers = [nn.Linear(input_dim, layer_widths[0]), activation()]
            for i in range(1, num_layers):
                w_in = layer_widths[i-1]
                w_out = layer_widths[i]
                layers.extend([nn.Linear(w_in, w_out), activation()])
            layers.append(nn.Linear(layer_widths[-1], num_out))
        if last_layer:
            layers.append(last_layer)
        self.model = nn.Sequential(*layers)

    def initialize(self):
        for layer in self.model[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight.data, gain=self.gain)
                nn.init.zeros_(layer.bias.data)
        final_layer = self.model[-1]
        nn.init.xavier_normal_(final_layer.weight.data, gain=1.0)
        nn.init.zeros_(final_layer.bias.data)

    def forward(self, data):
        # print(data.shape)
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        return self.model(data)


class MultipleMLPModel(nn.Module):
    def __init__(self, input_dim, layer_widths, num_models, activation=None,
                 last_layer=None):
        nn.Module.__init__(self)
        self.models = nn.ModuleList([MLPModel(
            input_dim, layer_widths, activation=activation,
            last_layer=last_layer, num_out=1) for _ in range(num_models)])
        self.num_models = num_models

    def forward(self, data):
        num_data = data.shape[0]
        data = data.view(num_data, -1)
        outputs = [self.models[i](data) for i in range(self.num_models)]
        return torch.cat(outputs, dim=1)
