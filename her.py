from collections import deque
import torch
import numpy as np
import copy
import time
class HER:
    def __init__(self):
        self.buffer=deque()
    def reset(self):
        self.buffer=deque()
    def keep(self,item):
        self.buffer.append(item)

    def backward(self):
        idx=0
        num=len(self.buffer)
        goal=self.buffer[-1][-2][:7]
        for i in range(num):
            temp=copy.deepcopy(self.buffer[-1-i][-2][:7])
            if (list(temp)==list(goal)):
                self.buffer[-1-i][2]=1.0 #reward
                self.buffer[-1-i][4]=0 #mask
                idx+=1
            self.buffer[-1-i][-2][13:]=copy.deepcopy(goal)
            self.buffer[-1-i][0][13:]=copy.deepcopy(goal)
            
        return self.buffer