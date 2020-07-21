import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(15, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x)) # -1 ~ 1
        
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(15, 128)
        self.fc2 = nn.Linear(128, 64)
        
        self.fc1_1 = nn.Linear(3, 64)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x1, a1):
        x1 = F.gelu(self.fc1(x1))
        x1 = self.fc2(x1)
        a1 = self.fc1_1(a1)
        x1a1 = torch.cat([x1,a1], dim = 1) # [32 + 32]
        x = F.relu(self.fc3(x1a1))
        x = self.fc4(x)
        
        return x