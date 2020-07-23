import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np

class Actor_net(nn.Module):
    def __init__(self):
        super(Actor_net, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        self.f_mean = nn.Linear(128, 4)
        self.f_std=nn.Linear(128,4)
        
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)
        nn.init.kaiming_normal_(self.f_mean.weight.data)
        nn.init.kaiming_normal_(self.f_std.weight.data)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x= F.gelu(self.fc3(x))

        mu = F.tanh(self.f_mean(x)) # -1 ~ 1
        std= F.softplus(self.f_std(x))
        return mu,std

class Critic_net(nn.Module):
    def __init__(self):
        super(Critic_net, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        self.f_value = nn.Linear(128, 1)
        
        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)
        nn.init.kaiming_normal_(self.f_value.weight.data)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x= F.gelu(self.fc3(x))

        out=self.f_value(x)
        
        return out
