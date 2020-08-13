import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal
import numpy as np
from torch.autograd import Variable
FLOAT = torch.FloatTensor

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.LSTMCell(256, 128)
        self.fc4 = nn.Linear(128, 4)
        
        self.L1=nn.LayerNorm(256)
        self.L2=nn.LayerNorm(256)
        self.L3=nn.LayerNorm(128)
        self.L4=nn.LayerNorm(4)

        self.cx=Variable(torch.zeros(1,128)).type(FLOAT).to('cuda')
        self.hx=Variable(torch.zeros(1,128)).type(FLOAT).to('cuda')

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        #nn.init.kaiming_normal_(self.fc3.weight.data)
        nn.init.kaiming_normal_(self.fc4.weight.data)

    def forward(self, x,hidden_state=None):
        x = self.fc1(x)
        x = F.gelu(self.L1(x))
        x = self.fc2(x)
        x = F.gelu(self.L2(x))
        
        if hidden_state==None:
            x=x.unsqueeze(0)
            hx, cx =self.fc3(x,(self.hx,self.cx))
            self.hx=hx
            self.cx=cx
        else:
            hx,cx=self.fc3(x,hidden_state)
        
        x=hx    
        x = self.fc4(x)
        x=F.tanh(self.L4(x)) # -1 ~ 1
        
        return x,(hx,cx)
    
    def reset(self,done=True):
        if done==True:
           self.cx=Variable(torch.zeros(1,128)).type(FLOAT).to('cuda')
           self.hx=Variable(torch.zeros(1,128)).type(FLOAT).to('cuda')
        else:
            self.cx=Variable(self.cx.data).type(FLOAT).to('cuda')
            self.hx=Variable(self.hx.data).type(FLOAT).to('cuda')


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(13, 256)
        self.fc2_1 = nn.Linear(256, 128)
        self.L1=nn.LayerNorm(256)

        nn.init.kaiming_normal_(self.fc1.weight.data)
        nn.init.kaiming_normal_(self.fc2_1.weight.data)
        
        #self.fc1_1=nn.LSTM(3,128,1)
        self.fc1_1 = nn.Linear(4, 256)
        self.L2=nn.LayerNorm(256)
        self.fc2_2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        nn.init.kaiming_normal_(self.fc1_1.weight.data)
        nn.init.kaiming_normal_(self.fc2_2.weight.data)
        nn.init.kaiming_normal_(self.fc3.weight.data)
        nn.init.kaiming_normal_(self.fc4.weight.data)

    def forward(self, x1, a1):
        x1 = self.fc1(x1)
        x1=F.gelu(self.L1(x1))
        x1 = F.gelu(self.fc2_1(x1))
        a1 = self.fc1_1(a1)
        a1=F.gelu(self.L2(a1))
        a1 = F.gelu(self.fc2_2(a1))
        x1a1 = torch.cat([x1,a1], dim = 1) # [32 + 32]
        x = F.gelu(self.fc3(x1a1))
        x = self.fc4(x)
        
        return x

class RND(nn.Module):
    def __init__(self):
        super(RND,self).__init__()
        
        self.p1=nn.Linear(13,128)
        self.p2=nn.Linear(128,128)
        
        self.t1=nn.Linear(13,128)
        self.t2=nn.Linear(128,128)

        #nn.init.kaiming_normal_(self.p1.weight.data)
        #nn.init.kaiming_normal_(self.p2.weight.data)
        nn.init.kaiming_normal_(self.t1.weight.data)
        nn.init.kaiming_normal_(self.t2.weight.data)

        for p in self.t1.parameters():
            p.requires_grad=False
        for p in self.t2.parameters():
            p.requires_grad=False

    def forward(self,x):
        ta=F.gelu(self.t1(x))
        ta=self.t2(ta)
        pr=F.gelu(self.p1(x))
        pr=self.p2(pr)
        return ta,pr