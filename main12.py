 
import argparse
import datetime
import gym
import pybullet as p
import numpy as np
import itertools
import torch
import time

from con_env8 import UR5_robotiq
from net10 import Actor, Critic
from collections import deque
from matplotlib import pyplot as plt
from epi_memory import EpisodicMemory
import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
FLOAT = torch.FloatTensor
device = torch.device("cuda")
print(device)

state_size=13
action_size=4
#Hyperparameters
learning_rate = 2e-4 #0.0005
gamma         = 0.99 #0.98
batch_size    = 128
alpha=0.2
tau=0.1

actor=Actor().to(device)
critic=Critic().to(device)
actor_target=Actor().to(device)
critic_target=Critic().to(device)

actor_target.load_state_dict(actor.state_dict())
critic_target.load_state_dict(critic.state_dict())

memory=EpisodicMemory(100000,150,window_length=1)
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


def to_tensor(ndarray, volatile=False, requires_grad=False, dtype=FLOAT):
    return Variable(
        torch.from_numpy(ndarray), volatile=volatile, requires_grad=requires_grad
    ).type(dtype)

def train():
    experiences = memory.sample(batch_size)
    if len(experiences) == 0: # not enough samples
        return

    policy_loss_total = 0
    value_loss_total = 0
    for t in range(len(experiences) - 1): # iterate over episodes
        target_cx = Variable(torch.zeros(batch_size, 128)).type(FLOAT).to(device)
        target_hx = Variable(torch.zeros(batch_size, 128)).type(FLOAT).to(device)

        cx = Variable(torch.zeros(batch_size, 128)).type(FLOAT).to(device)
        hx = Variable(torch.zeros(batch_size, 128)).type(FLOAT).to(device)

            # we first get the data out of the sampled experience
        state0 = np.stack((trajectory.state0 for trajectory in experiences[t]))
            # action = np.expand_dims(np.stack((trajectory.action for trajectory in experiences[t])), axis=1)
        action = np.stack((trajectory.action for trajectory in experiences[t]))
        reward = np.expand_dims(np.stack((trajectory.reward for trajectory in experiences[t])), axis=1)
            # reward = np.stack((trajectory.reward for trajectory in experiences[t]))
        state1 = np.stack((trajectory.state0 for trajectory in experiences[t+1]))

        state0 = torch.Tensor(state0).to(device)
        action = torch.Tensor(action).to(device)
        reward = torch.Tensor(reward).to(device)
        state1 = torch.Tensor(state1).to(device)

        target_action, (target_hx, target_cx) = actor_target(state1, (target_hx, target_cx))
        next_q_value = critic_target(state1,target_action)
        next_q_value=next_q_value.detach()

        target_q = reward+ gamma*next_q_value

            # Critic update
        MSE = torch.nn.MSELoss()
        current_q = critic(state0,action)
        value_loss = MSE(current_q, target_q)
        value_loss /= len(experiences) # divide by trajectory length
        #value_loss_total += value_loss

            # Actor update
        action_n, (hx, cx) = actor(state0, (hx, cx))
        policy_loss = -critic(state0, action_n).mean()
        policy_loss /= len(experiences) # divide by trajectory length
        #policy_loss_total += policy_loss.mean()
    
        # backward.
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()
    
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()
    
    # soft target update
    soft_update(actor, actor_target)
    soft_update(critic, critic_target)

def main():
    flag = 0
    flag2 = 0
    print_interval = 20 
    score = 0.0  
    global_step=0
    succ= []
    F = []
    epi=[]
    avg1=[]
    avg2=[]
    avg3=[]
    error=[]
    flag2=0
    sigma=0.8
    min_sigma=0.05
    for epi_n in range(3000):
        state = env.reset()
        #pre_noise=np.zeros(action_size)
        done = False
        step = 0
        score = 0
        if epi_n > 1000 and epi_n%20==0:
            torch.save(actor,"./saved_model44/model"+str(epi_n)+".pth")
        while not done:
            step += 1
            global_step+=1
            action, _ = actor(torch.FloatTensor(state).to(device))
            action=action[0]
            #noise=ou_noise(pre_noise,action_size,sigma)
            if sigma>min_sigma:
                noise=my_noise(action.detach().cpu().numpy(),action_size,sigma)
            
            else:
                noise=gaussian_noise(action_size,sigma)
            action=(action.cpu()+torch.Tensor(noise)).clamp(-1.0,1.0)
            #print(noise,action)
            next_state, reward, done, info= env.step(list(action))
            a=action.detach().numpy()
            
            memory.append(state,a,reward,done)
            #memory.append((state, a, reward, next_state,  mask))
            if epi_n>145:
                actor.reset(done=done)
                train()
            score = reward + gamma*score
            state = next_state

            if step>150 or done==True:
                #actor.reset(done=True)
                if reward>2:
                    succ.append(1)
                else:
                    succ.append(0)
                if sigma>min_sigma:
                    sigma*=0.995
                else:
                    sigma=min_sigma
                    flag+=1
                
                if flag==1 and flag2==0:
                    sigma=0.4
                    flag2=1
                F.append(score)
                error.append(info)
                epi.append(epi_n)
                print('n_episode: ',epi_n,'score: ',score,'step: ',step,'noise: ',sigma)
                if epi_n>500:
                    print('error: ',info)
                break
        if epi_n%10==0 and epi_n >0:
            avg1.append(sum(F)/len(F))
            F=[]
            avg2.append(sum(succ)/len(succ))
            succ=[]
            avg3.append(sum(error)/len(error))
            error=[]


    plt.subplot(221)
    plt.plot(avg1)
    plt.subplot(222)
    plt.plot(avg2)
    plt.subplot(223)
    plt.plot(avg3)
    plt.show()
                   
    env.close()

if __name__ == '__main__':
    main()

end = True
while end:
    end = True