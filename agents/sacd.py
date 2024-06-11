import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, input_dim, n_action, fc1=256, fc2=512):
        super(Actor, self).__init__()
        self.fcl_1 = nn.Linear(input_dim, fc1)
        self.fcl_2 = nn.Linear(fc1, fc2)
        self.fcl_3 = nn.Linear(fc2, fc1)
        self.fcl_4 = nn.Linear(fc1, n_action)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):
        x = F.relu(self.fcl_1(state))
        x = F.relu(self.fcl_2(x))
        x = F.relu(self.fcl_3(x))
        x = F.relu(self.fcl_4(x))
        action_probs = self.softmax(x)
        return action_probs
    

    def evaluate(self, state, epsilon=1e-6):
        action_probs = self.forward(state)

        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)

        z = action_probs == 0.0
        z = z.float() * 1e-8
        log_action_probabilities = torch.log(action_probs + z)
        return action.detach().cpu(), action_probs, log_action_probabilities
    

    def get_det_action(self, state):
        action_probs = self.forward(state)
        dist = Categorical(action_probs)
        action = dist.sample().to(state.device)
        return action.detach().cpu()
    


class Critic(nn.Module):
    def __init__(self, input_dim, n_actions, fc1, fc2, seed=1):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcl_1 = nn.Linear(input_dim, fc1)
        self.fcl_2 = nn.Linear(fc1, fc2)
        self.fcl_3 = nn.Linear(fc2, fc1)
        self.fcl_4 = nn.Linear(fc1, n_actions)
        self.register_parameter()

    def reset_parameters(self):
        self.fcl_1.weight.data.uniform_(*hidden_init(self.fcl_1))
        self.fcl_2.weight.data.uniform_(*hidden_init(self.fcl_2))
        self.fcl_3.weight.data.uniform_(*hidden_init(self.fcl_3))
        self.fcl_4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = F.relu(self.fcl_1(state))
        x = F.relu(self.fcl_2(x))
        x = F.relu(self.fcl_3(x))
        return self.fcl_4(x)
    
    

