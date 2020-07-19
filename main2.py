 
import argparse
import datetime
import gym
import pybullet as p
import numpy as np
import itertools
import torch
import time

from dis_env4 import UR5_robotiq
from net2 import DQN
#from prioritized_memory import Memory
from Per import *
from matplotlib import pyplot as plt

import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Allegro",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--GUI', default="GUI",
                    help='Pybullet physics server mode  GUI | DIRECT (default: GUI)')
# parser.add_argument('--GUI', default="DIRECT",
                    # help='Pybullet physics server mode  GUI | DIRECT (default: GUI)')
parser.add_argument('--suffix', default="00",
                    help='Disticguish models (default: 00)')
parser.add_argument('--dist_threshold', type=float, default=0.01,
                    help='Goal and State distance threshold (default: 0.03)')
parser.add_argument('--future_k', type=int, default=4,
                    help='future_k (default: 4)')
parser.add_argument('--epi_step', type=int, default=50,
                    help='Max_episode_steps (default: 50)')
parser.add_argument('--evaluate', type=bool, default=False,
                    help='evaluate (default: False)')

args = parser.parse_args()

state_size=11
action_size=9
#Hyperparameters
learning_rate = 0.0005 #0.0005
gamma         = 0.5  #0.98
batch_size    = 128
alpha=0.2
soft_tau=0.1

model=DQN()

target_model=DQN()
target_model.load_state_dict(model.state_dict())


memory=Memory(100000)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Environment
env = UR5_robotiq(args)
# env.getCameraImage()

pose = env.getRobotPose()
print("Robot initial pose")
print(pose)

pose = env.holePos
pose.extend(env.Orn)
#pose[2] += 0.15 
print("Robot target pose")
print(env.holePos)
# p.addUserDebugText('O',np.array(env.holePos)[:3])
# env.moveL(env.holePos)
# get = env.getRobotPose()
# print(get)

# torch.cuda.device(0)
# cuda = torch.device('cuda')
'''

pose = env.getRobotPose()
print("Robot final pose")
print(pose)
'''
   
def action_select(y):

    if y == 0 :
        action = [0.001,0,0,0]
    elif  y == 1 :
        action = [-0.001,0,0,0]
    elif  y == 2 :
        action = [0,0.001,0,0]
    elif  y == 3 :
        action = [0,-0.001,0,0]
    elif  y == 4 :
        action = [0,0,0.001,0] 
    elif  y == 5 :
        action = [0,0,-0.001,0]
    elif y==6:
        action = [0,0,0,0.0005]
    elif y==7:
        action = [0,0,0,-0.0005]
    elif y==8:
        action = [0,0,-0.005,0]
    
    return action     

def append_sample(state, action, reward, next_state, done):
    target = model(Variable(torch.FloatTensor(state))).data
    old_val = target[0][action]
    target_val = target_model(Variable(torch.FloatTensor(next_state))).data
    next_val = model(Variable(torch.FloatTensor(next_state))).data
    temp=torch.max(next_val, 1)[1].unsqueeze(1)
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + gamma *target_val[0][temp]

    error = abs(old_val - target[0][action])

    memory.add(error, (state, action, reward, next_state, done))

def update_target():
    target_model.load_state_dict(model.state_dict())

def train_model():
    tree_idx,mini_batch= memory.sample(batch_size)
    #mini_batch, idxs, is_weights = memory.sample(batch_size)
    mini_batch = np.array(mini_batch).transpose()

    try:
        states = np.vstack(mini_batch[0])
    except:
        pass
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.vstack(mini_batch[3])
    dones = mini_batch[4]

        # bool to binary
    dones = dones.astype(int)

        # Q function of current state
    states = torch.Tensor(states)
    states = Variable(states).float()
    pred = model(states)
        # one-hot encoding
    a = torch.LongTensor(actions).view(-1, 1)

    one_hot_action = torch.FloatTensor(batch_size, action_size).zero_()
    one_hot_action.scatter_(1, a, 1)

    pred = torch.sum(pred.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
    next_states = torch.Tensor(next_states)
    next_states = Variable(next_states).float()
    next_pred = target_model(next_states).data
    
    next_val = model(Variable(torch.FloatTensor(next_states))).data
    temp=torch.max(next_val, 1)[1].squeeze(-1).view(-1,1)
    one_hot_action2 = torch.FloatTensor(batch_size, action_size).zero_()
    one_hot_action2.scatter_(1, temp, 1)

    next_pred = torch.sum(next_pred.mul(Variable(one_hot_action2)), dim=1)
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)
        # Q Learning: get maximum Q value at s' from target model
    target = rewards + (1 - dones) * gamma * next_pred
    target = Variable(target)

    errors = torch.abs(pred - target).data.numpy()

        # update priority
    #for i in range(batch_size):
    #    idx = idxs[i]
   #     memory.update(idx, errors[i])
    #print(tree_idx,errors)
    memory.batch_update(tree_idx,errors)

    optimizer.zero_grad()
    #loss = (torch.FloatTensor(is_weights) * F.mse_loss(pred, target)).mean()
    loss = (F.mse_loss(pred, target)).mean()
    loss.backward()
    optimizer.step()


def main():
    save_action = []
    flag = 0
    flag2 = 0
    print_interval = 20 
    score = 0.0  
    global_step=0
    last = []
    F = []
    epi=[]
    ee=1.0
    ep_min=0.05
    for epi_n in range(5000):
        state = env.reset()
        done = False
        step = 0
        score = 0
        if epi_n > 2000 and epi_n%20==0:
            torch.save(model,"./saved_model2/model_policy_"+str(epi_n)+".pth")
        while not done:
            step += 1
            global_step+=1
            action = model.act(torch.FloatTensor(state),ee)
            #print(state,action)   #torch.from_numpy(s) > s  
            next_state, reward, done, _= env.step(action_select(action))
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            #append_sample(state, action, reward, next_state, done)
            if reward>-10:
                memory.store((state,action,reward,next_state,done))
                score += reward
            state = next_state

            if global_step >= 150:
                train_model()
            
            if step>500:
                done=True

            if done:
                update_target()
                if ee>ep_min:
                    ee*=0.995
                    #print(ee)
                else:
                    ee=ep_min
                flag += 1
                F.append(score)
                epi.append(epi_n)
                print('n_episode: ',epi_n,'score: ',score,'step: ',step,'epsilon: ',ee)
                break 
            if flag == 20:
                flag2 += 1
            flag = 0 
        
    plt.plot(epi,F)   
    
    plt.savefig('foo.png')
                   
    env.close()

if __name__ == '__main__':
    main()

end = True
while end:
    end = True