 
import argparse
import datetime
import gym
import pybullet as p
import numpy as np
import itertools
import torch
import time

from con_env2 import UR5_robotiq
from net5 import Actor, Critic
from collections import deque
from matplotlib import pyplot as plt
from Per import *
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

state_size=8
action_size=4
#Hyperparameters
learning_rate = 5e-6 #0.0005
gamma         = 0.8  #0.98
batch_size    = 256
alpha=0.2
tau=0.1

actor=Actor()
critic=Critic()
actor_target=Actor()
critic_target=Critic()

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

memory=Memory(10000)
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
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
device = torch.device('cpu')
#0.01 
def append_sample(state, action, reward, next_state, mask):
    q_value = critic(states, actions).squeeze(1)
    
    target = reward + mask * gamma * critic_target(next_state, actor_target(next_state)).squeeze(1)

    error = abs(old_val - target[0][action])

    memory.add(error, (state, action, reward, next_state, mask))

def soft_update(net,target_net):
    for param, target_param in zip(net.parameters(),target_net.parameters()):
        target_param.data.copy_(tau*param.data+(1.0-tau)*target_param.data)

def ou_noise(x,dim,sigma):
    rho=0.07
    mu=0
    dt=1e-2
    return x-rho*x*dt+sigma*np.sqrt(dt)*np.random.normal(size=dim)

def gaussian_noise(dim,sigma):
    return sigma*np.random.normal(size=dim)

def my_noise(x,dim,sigma):
    return x*sigma*np.random.normal(size=dim)-x*sigma*3

def train():
    #random_mini_batch=random.sample(memory,batch_size)
    tree_idx,mini_batch= memory.sample(batch_size)
    mini_batch = np.array(mini_batch).transpose()
    # data 분배
    #mini_batch = np.array(random_mini_batch) 
    #states = np.vstack(mini_batch[:, 0]) 
    #actions = list(mini_batch[:, 1])
    #rewards = list(mini_batch[:, 2])
    #next_states = np.vstack(mini_batch[:, 3])
    #masks = list(mini_batch[:, 4]) 
    states = np.vstack(mini_batch [0]) 
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.vstack(mini_batch[3])
    masks = list(mini_batch[4]) 

    # tensor.
    states = torch.Tensor(states)
    actions = torch.Tensor(actions)
    rewards = torch.Tensor(rewards) 
    next_states = torch.Tensor(next_states)
    masks = torch.Tensor(masks)
    # actor loss
    actor_loss = -critic(states, actor(states)).mean()
    #critic loss
    MSE = torch.nn.MSELoss()

    target = rewards + masks * gamma * critic_target(next_states, actor_target(next_states)).squeeze(1)
    q_value = critic(states, actions).squeeze(1)
    critic_loss = MSE(q_value, target.detach())
    
    errors=torch.abs(q_value-target.detach()).data.numpy()
    memory.batch_update(tree_idx,errors)
    
    # backward.
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # soft target update
    soft_update(actor, actor_target)
    soft_update(critic, critic_target)

def main():
    save_action = []
    flag = 0
    flag2 = 0
    print_interval = 20 
    score = 0.0  
    global_step=0
    succ= []
    F = []
    epi=[]
    sigma=0.8
    min_sigma=0.01
    for epi_n in range(5000):
        state = env.reset()
        #pre_noise=np.zeros(action_size)
        done = False
        step = 0
        score = 0
        if epi_n > 2000 and epi_n%20==0:
            torch.save(actor,"./saved_model/model"+str(epi_n)+".pth")
        while not done:
            step += 1
            global_step+=1
            
            action = actor(torch.FloatTensor(state))
            #print(action)
            #noise=ou_noise(pre_noise,action_size,sigma)
            noise=my_noise(action.detach().numpy(),action_size,sigma)
            action=(action+torch.Tensor(noise)).clamp(-1.0,1.0)
            #print(noise,action)
            next_state, reward, done, _= env.step(list(action))
            mask =0  if done else 1
            a=action.detach().numpy()
            
            memory.store((state,a,reward,next_state,mask))
            #memory.append((state, a, reward, next_state,  mask))
            if global_step>300:
                train()
            if epi_n>200:
                print(action)
                print(next_state)
            score += reward
            state = next_state
            pre_noise=noise

            if step>50:
                done=True
            if done:
                if reward>10:
                    succ.append(1)
                else:
                    succ.append(0)
                if sigma>min_sigma:
                    sigma*=0.995
                else:
                    sigma=min_sigma
                flag += 1
                F.append(float(score/(step-1)))
                epi.append(epi_n)
                print('n_episode: ',epi_n,'score: ',float(score/(step-1)),'step: ',step,'noise: ',sigma)
                break 
            if flag == 20:
                flag2 += 1
            flag = 0 
        
    
    plt.plot(epi,F)   
    f = open("saved_model/fig.txt", 'w')
    f.write(str(succ))
    f.close()
    plt.show()
                   
    env.close()

if __name__ == '__main__':
    main()

end = True
while end:
    end = True