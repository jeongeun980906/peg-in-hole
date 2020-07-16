import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
    def sample_action(self, obs, epsilon):
        self.out = self.forward(obs).detach().cpu().numpy()
        
        coin = random.random()
    
        if coin < epsilon:
            return random.randint(0,7)        #0 -> -1 random.randint(-1,1)-> x
        else :         #-0.1or0.1 randompick 이후 pose에 random하게 합
            return np.argmax(self.out)    #out.argmax().item() -> x
