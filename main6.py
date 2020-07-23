 
import argparse
import datetime
import gym
import pybullet as p
import numpy as np
import itertools
import torch
import time

from con_env2 import UR5_robotiq
from net6 import Critic_net,Actor_net
from collections import deque
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

state_size=8
action_size=4

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
class Actor(nn.Module):
    def __init__(self,lr_rate,ratio_clipping):
        super(Actor,self).__init__()

        self.std_bound=[1e-2,1.]
        self.actor_network=Actor_net()
        self.optimizer=torch.optim.Adam(self.actor_network.parameters(),lr=lr_rate)
        self.ratio_clipping=ratio_clipping
        
    def log_pdf(self,mu,std,action):
        std=torch.clamp(std,min=self.std_bound[0],max=self.std_bound[1])
        var=std**2
        log_policy_pdf=-0.5*(action-mu)**2/var-0.5*torch.log(var*2*np.pi)
        return torch.sum(log_policy_pdf,dim=1,keepdim=True)
    
    def get_policy_action(self,state):
        self.actor_network.eval()
        with torch.no_grad():
            mu_a,std_a=self.actor_network(state.view(1,state_size))
            mu_a,std_a=mu_a[0],std_a[0]
            std_a=torch.clamp(std_a,self.std_bound[0],self.std_bound[1])
            action=torch.normal(mu_a,std_a)
        return mu_a,std_a,action
    
    def update(self,states,actions,advantages,log_old_policy_pdf):
        self.actor_network.train()
        mu_a,std_a=self.actor_network(states)
        log_policy_pdf=self.log_pdf(mu_a,std_a,actions)
        ratio=torch.exp(log_policy_pdf-log_old_policy_pdf)
        clipped_ratio=torch.clamp(ratio,1.-self.ratio_clipping,1.-self.ratio_clipping)
        
        surrogate=-torch.min(ratio*advantages.detach(),clipped_ratio*advantages.detach())
        loss=torch.mean(surrogate)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Critic(nn.Module):
    def __init__(self,lr_rate):
        super(Critic,self).__init__()
        self.critic_network=Critic_net()
        self.optimizer=torch.optim.Adam(self.critic_network.parameters(),lr=lr_rate)
        
    def get_value(self,states):
        self.critic_network.eval()
        values=self.critic_network(states)
        return values
    def update(self,states,targets):
        self.critic_network.train()
        values=self.critic_network(states)
        loss=F.mse_loss(values,targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self):
        self.gamma=0.95
        self.gae_lambda=0.9
        self.batch_size=64
        self.epochs=10
        
        self.actor_lr=0.0001
        self.critic_lr=0.001
        self.ratio_clipping=0.2

        self.env=env
        self.state_dim=state_size
        self.action_dim=action_size

        self.actor=Actor(self.actor_lr,self.ratio_clipping)
        self.critic=Critic(self.critic_lr)
        
        self.save_epi_reward=[]
        
    
    def gae_target(self,rewards,v_values,next_v_value,done):
        n_step_targets=torch.zeros_like(rewards)
        gae=torch.zeros_like(rewards)
        gae_cumulative=0.
        forward_val=0.
        if not done:
            forward_val=next_v_value
        for k in reversed(range(0,len(rewards))):
            delta=rewards[k]+self.gamma*forward_val-v_values[k]
            gae_cumulative=self.gamma*self.gae_lambda*gae_cumulative+delta
            gae[k]=gae_cumulative
            forward_val=v_values[k]
            n_step_targets[k]=gae[k]+v_values[k]
        return gae,n_step_targets
    
    def unpack_batch(self,batch):
        unpack=[]
        for idx in range(len(batch)):
            unpack.append(batch[idx])
        unpack=torch.cat(unpack,axis=0)
        return unpack
    
    def train(self,max_episode_num):
        batch_state,batch_action,batch_reward=[],[],[]
        batch_log_old_policy_pdf=[]
        
        for episode in range(max_episode_num):
            time,episode_reward,done=0,0,False
            state=self.env.reset()
            state = torch.from_numpy(state).type(torch.FloatTensor)
            
            while not done:
                mu_old,std_old,action=self.actor.get_policy_action(state)
                action=action.detach().numpy()
                mu_old=mu_old.detach().numpy()
                std_old=std_old.detach().numpy()
                
                var_old=std_old**2
                log_old_policy_pdf=-0.5*(action-mu_old)**2/var_old-0.5*np.log(var_old*2*np.pi)
                log_old_policy_pdf=np.sum(log_old_policy_pdf)
                
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.from_numpy(next_state).type(torch.FloatTensor)
                action=torch.from_numpy(action).type(torch.FloatTensor)
                reward=torch.FloatTensor([reward])
                log_old_policy_pdf = torch.FloatTensor([log_old_policy_pdf])
                
                state      = state.view(1, self.state_dim)
                next_state = next_state.view(1, self.state_dim)
                action     = action.view(1, self.action_dim)
                reward     = reward.view(1, 1)
                log_old_policy_pdf = log_old_policy_pdf.view(1,1)
                
                batch_state.append(state)
                batch_action.append(action)
                batch_reward.append((reward+8)/8)
                batch_log_old_policy_pdf.append(log_old_policy_pdf)
                
                if len(batch_state)<self.batch_size:
                    state=next_state[0]
                    episode_reward+=reward[0]
                    time+=1
                    continue
                    
                states=self.unpack_batch(batch_state)
                actions=self.unpack_batch(batch_action)
                rewards = self.unpack_batch(batch_reward)
                log_old_policy_pdfs = self.unpack_batch(batch_log_old_policy_pdf)
                batch_state, batch_action, batch_reward = [],[],[]
                batch_log_old_policy_pdf = []
                
                v_values=self.critic.get_value(states)
                next_v_value=self.critic.get_value(next_state)
                gae,y_i=self.gae_target(rewards,v_values,next_v_value,done)
                
                for _ in range(self.epochs):
                    self.actor.update(states,actions,gae,log_old_policy_pdfs)
                    self.critic.update(states,y_i)
                    
                state = next_state[0]
                episode_reward += reward[0]
                time += 1
                
                if step>50:
                    done=True

            self.save_epi_reward.append(episode_reward.item())
            if len(self.save_epi_reward) < 20:
                print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(self.save_epi_reward))
            else:
                print('Episode:', episode+1, 'Time:', time, 'Reward(ave of recent20):', np.mean(self.save_epi_reward[-20:]))
    
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()

def main():
    
    MAX_EPISODE = 3000
    
    agent=Agent()
    agent.train(MAX_EPISODE)
    agent.plot_result()
                   
    env.close()

if __name__ == '__main__':
    main()

end = True
while end:
    end = True