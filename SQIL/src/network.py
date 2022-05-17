import torch
import torch.nn.functional as F
import math
import random
from torch import nn
from torch.distributions import Categorical

class SoftQNetwork(nn.Module):
    def __init__(self, input_shape, output_dim, alpha):
        super(SoftQNetwork, self).__init__()
        self.alpha = alpha
        n_input_channels = input_shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, *input_shape)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        return self.linear(self.cnn(x))

    def getV(self, q_value):
        v = self.alpha * torch.log(torch.sum(torch.exp(q_value/self.alpha), dim=1, keepdim=True))
        return v
        
    def choose_action(self, state, epsilon):
        state = torch.FloatTensor(state)
        # print('state : ', state)
        with torch.no_grad():
            q = self.forward(state)
            v = self.getV(q).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            if epsilon < random.uniform(0, 1):
                a = torch.argmax(dist)
            else:
                c = Categorical(dist)
                a = c.sample()
        return a.item()