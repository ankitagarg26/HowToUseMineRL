import torch
import torch.nn.functional as F
import math
import numpy as np
import random
from torch import nn
from torch.distributions import Categorical



def init_lecun_normal(tensor, scale=1.0):
    """Initializes the tensor with LeCunNormal."""
    fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)

@torch.no_grad()
def init_chainer_default(layer):
    """Initializes the layer with the chainer default.
    weights with LeCunNormal(scale=1.0) and zeros as biases
    """
    assert isinstance(layer, nn.Module)

    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        init_lecun_normal(layer.weight)
        if layer.bias is not None:
            # layer may be initialized with bias=False
            nn.init.zeros_(layer.bias)
    return layer

def constant_bias_initializer(bias=0.0):
    @torch.no_grad()
    def init_bias(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            m.bias.fill_(bias)

    return init_bias
    
class MLP(nn.Module):
    """Multi-Layer Perceptron"""

    def __init__(
        self, in_size, out_size, hidden_sizes, nonlinearity=F.relu, last_wscale=1
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_sizes = hidden_sizes
        self.nonlinearity = nonlinearity
        super().__init__()
        if hidden_sizes:
            self.hidden_layers = nn.ModuleList()
            self.hidden_layers.append(nn.Linear(in_size, hidden_sizes[0]))
            for hin, hout in zip(hidden_sizes, hidden_sizes[1:]):
                self.hidden_layers.append(nn.Linear(hin, hout))
            self.hidden_layers.apply(init_chainer_default)
            self.output = nn.Linear(hidden_sizes[-1], out_size)
        else:
            self.output = nn.Linear(in_size, out_size)

        init_lecun_normal(self.output.weight, scale=last_wscale)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        h = x
        if self.hidden_sizes:
            for layer in self.hidden_layers:
                h = self.nonlinearity(layer(h))
        return self.output(h)

class SoftQNetwork(nn.Module):    
    def __init__(self, n_actions, n_input_channels=4, activation=F.relu, bias=0.1):
        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation

        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # Modified from 3136 -> 1024
        self.a_stream = MLP(1024, n_actions, [512])
        self.v_stream = MLP(1024, 1, [512])

        self.conv_layers.apply(init_chainer_default)  # MLP already applies
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]
        h = h.reshape(batch_size, -1)
        ya = self.a_stream(h)
        mean = torch.reshape(torch.sum(ya, dim=1) / self.n_actions, (batch_size, 1))
        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h)

        ya, ys = torch.broadcast_tensors(ya, ys)
        q = ya + ys
        return q
    
    def getV(self, q_values):
        q_max = torch.broadcast_tensors(q_values.max(dim=-1, keepdim=True)[0],q_values)[0]
        q_soft = (q_max[:, 0] + (q_values - q_max).exp().sum(dim=-1).log())
        
        return q_soft
    
    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state)
        with torch.no_grad():
            q = self.forward(state)
            a = torch.argmax(q)
        return a.item()