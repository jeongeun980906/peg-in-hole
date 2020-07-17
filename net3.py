import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()

        self.feature = nn.Linear(17,128)
        
        self.a1 =nn.Linear(128, 128)
        self.a2=nn.Linear(128, 9)
        
        
        self.v1 =nn.Linear(128, 128)
        self.v2=nn.Linear(128, 1)
        
        
    def forward(self, x):
        x = F.gelu(self.feature(x))
        advantage = F.gelu(self.a1(x))
        advantage=self.a2(advantage)
        value     = F.gelu(self.v1(x))
        value=self.v2(value)
        return value + advantage  - advantage.mean()
    
    def act(self, state, epsilon):
        seed=random.random()
        if seed > epsilon:
            state   = Variable(torch.FloatTensor(state).unsqueeze(0)).detach()
            q_value = self.forward(state).view(-1,9)
            #print(torch.max(q_value[0],0)[1])
            action  = torch.max(q_value[0],0)[1]
            #print(q_value,action)
        else:
            action = random.randrange(9)
        return int(action)