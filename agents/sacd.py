import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque, namedtuple
import random



""" Replay - Buffer """

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device):
        self.device = device
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experience = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experience if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experience if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)



"""  Actor and Critic Network """
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
    


