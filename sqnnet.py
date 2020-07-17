import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
init_w=3e-3
        
        
class SoftQNetwork(nn.Module):
    def __init__(self):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(14, 128)
        #self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 128)
        self.linear4 = nn.Linear(128, 7)
        self.linear4.weight.data.uniform_(-init_w, init_w)
        self.linear4.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x=F.gelu(self.linear1(state))
        #x = F.gelu(self.linear2(x))
        x = F.gelu(self.linear3(x))
        x=self.linear4(x)
        return x
    
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)#.to('cuda')
        q_value = self.forward(state)
        policy= F.softmax(q_value).clamp(max=1-1e-20,min=1e-20)
        action = torch.multinomial(policy[0],1)
        #action  = action.detach().cpu().numpy()
        #policy  = policy.detach().numpy()
        return action[0]
        
    def entropy(self,state):
        state = state.unsqueeze(0)#.to('cuda')
        q_value =self.forward(state)
        policy=F.softmax(q_value[0]).clamp(max=1-1e-20,min=1e-20)
        entropy=-torch.sum(policy*torch.log(policy),dim=-1)
        policy  = policy.detach()
        entropy  = entropy.detach()
        return entropy
