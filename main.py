# -*- coding: utf-8 -*- 

import argparse
import datetime
import gym
import pybullet as p
import pybulletgym
import numpy as np
import itertools
import torch
import time
from sac import SAC
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
from ur5_robotiq85_env import UR5_robotiq
from Qnet import Qnet
from ReplayBuffer import ReplayBuffer
from matplotlib import pyplot as plt

import collections
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
Eval = False
#Hyperparameters
learning_rate = 0.0005 #0.0005
gamma         = 0.98  #0.98
batch_size    = 128

def create_state(state_dict):
    observation = state_dict["observation"]
    desired_goal = state_dict["desired_goal"]
    return np.concatenate((observation, desired_goal))


# Environment
env = UR5_robotiq(args)
# env.getCameraImage()

pose = env.getRobotPose()
print("Robot initial pose")
print(pose)

pose = env.holePos
pose.extend(env.Orn)
pose[2] += 0.15 
print("Robot target pose")
print(env.holePos)
# p.addUserDebugText('O',np.array(env.holePos)[:3])
# env.moveL(env.holePos)
# get = env.getRobotPose()
# print(get)

device = torch.device("cuda" if args.cuda else "cpu")
print(device)
# torch.cuda.device(0)
# cuda = torch.device('cuda')
'''

pose = env.getRobotPose()
print("Robot final pose")
print(pose)
'''
   
def action_select(y):

    if y == 0 :
        action = [0.008,0,0,0,0,0]
    elif  y == 1 :
        action = [-0.008,0,0,0,0,0]
    elif  y == 2 :
        action = [0,0.008,0,0,0,0]
    elif  y == 3 :
        action = [0,-0.008,0,0,0,0]
    elif  y == 4 :
        action = [0,0,0.008,0,0,0] 
    elif  y == 5 :
        action = [0,0,-0.008,0,0,0] 
    elif  y == 6 :
        action = [0,0,0,0,0,0]      
    elif  y == 7 :
        action = [0,0,0,0,0,0.013089967] #z축 3도 이동0.013089967 0.001090278
    else :
        action = [0,0,0,0,0,-0.013089967] #z축 -3도 이동0.026179933
    
    return action     

def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        
        q_out = q(s)
        q_a = q_out.gather(1,a)
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    q = Qnet().to(device)
    if Eval :
        q.load_state_dict(torch.load("models/DQN_{}_{}".format("UR5_zori", "06")))
    q_target = Qnet().to(device)
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()
    save_action = []
    flag = 0
    flag2 = 0
    print_interval = 20 
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)
    last = []
    F = []
    epi=[]
    for epi_n in range(5000):
        epsilon = max(0.01, 0.80 - 0.01*(epi_n/50)) #Linear annealing from 8% to 1%
        s = env.reset()
        done = False
        step = 0
        score = 0
        
        if epi_n > 2000 and epi_n%20==0:
           torch.save(q.state_dict(), "models/DQN_{}_{}".format("UR5_zori", "07"))

        for i in range(120):
            step += 1
            a = q.sample_action(torch.FloatTensor(s).to(device), epsilon)   #torch.from_numpy(s) > s  
            
            last.append(a)
            
            targetpose = env.getRobotPoseE()
            
            for i in range(6):
                targetpose[i] += action_select(a)[i]

            # print('action,pose')
            # print(action_select(a))
            # print(targetpose)

            s_prime, r, done, info = env.step(targetpose)
            # if epi_n > 600:
            #     time.sleep(0.1)
            #p.addUserDebugText('.',targetpose[0:3])
            done_mask = 0.0 if done else 1.0
            memory.put((s,a,float(r),s_prime, done_mask))
            
            s = s_prime
            score += r
            
        if done:
            flag += 1
        F.append(score)
        epi.append(epi_n)        
        
            
        if memory.size()>3000 and (not Eval):
            train(q, q_target, memory, optimizer)
        
        if epi_n%print_interval==0 and epi_n!=0:
            q_target.load_state_dict(q.state_dict())
            print("epi_n :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            epi_n, score/print_interval, memory.size(), epsilon*100))
            print(flag)
            
            
            if flag == 20:
                flag2 += 1
            flag = 0 
        # if flag2 >= 20:
        #     plt.xlim(-np.pi, np.pi)
        #     plt.ylim(-2.0, 2.0)
        #     plt.show()
        #     env.close()
        
        # if flag2 >= 20:
        #     end = True
        #     while end:
        #         env.reset()
                
        #         for i in range(len(save_action)):
                    
        #             targetpose = env.getRobotPose()
                
        #             for j in range(3):
        #                 targetpose[j] += action_select(save_action[i])[j]

        #             env.moveL(targetpose)
        #             time.sleep(0.1)
        #         print(env.getRobotPose())
        
    plt.plot(epi,F)   
    
    plt.savefig('foo.png')
                   
    env.close()

if __name__ == '__main__':
    main()

end = True
while end:
    end = True

##################################################################################

# torch.manual_seed(args.seed)
# np.random.seed(args.seed)
# # env.seed(args.seed)

# # Agent
# agent = SAC(env.observation_space, env.action_space, args)

# # TesnorboardX
# writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
#                                                              args.policy, "autotune" if args.automatic_entropy_tuning else "",args.suffix))

# # Memory with HER
# memory = ReplayMemory(args.replay_size, args.future_k)

# # Training Loop
# total_numsteps = 0
# updates = 0

# for i_episode in itertools.count(1):
#     episode_reward = 0
#     episode_steps = 0
#     done = False
#     # env.render()
#     # state = env.reset()
#     # print(state)
#     state_dict = env.reset()
#     memory.reset(state_dict)
#     while not done:
#         state = create_state(state_dict)
#         if args.start_steps > total_numsteps:
#             action = env.sample_action()  # Sample random action
#         else:
#             action = agent.select_action(state)  # Sample action from policy
#         # print("AA")
#         # print(len(her_memory))
#         if len(memory) > 10*env._max_episode_steps:

#             # Number of updates per step in environment
#             for i in range(args.updates_per_step):

#                 # Update parameters of all the networks
#                 critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

#                 writer.add_scalar('loss/critic_1', critic_1_loss, updates)
#                 writer.add_scalar('loss/critic_2', critic_2_loss, updates)
#                 writer.add_scalar('loss/policy', policy_loss, updates)
#                 writer.add_scalar('loss/entropy_loss', ent_loss, updates)
#                 writer.add_scalar('entropy_temprature/alpha', alpha, updates)
#                 updates += 1

#         next_state_dict, reward, done, _ = env.step(action) # Step
        
        
#         episode_steps += 1
#         total_numsteps += 1
#         episode_reward += reward

#         # Ignore the "done" signal if it comes from hitting the time horizon.
#         # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#         mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        

#         memory.store_transition(state_dict, action, reward, next_state_dict, mask) # Append transition to memory

#         state_dict = next_state_dict

#         if episode_steps == env._max_episode_steps or done:
#             break
    
#     memory.store_episode()

#     if total_numsteps > args.num_steps:
#         break

#     writer.add_scalar('reward/train', episode_reward, i_episode)
#     print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

#     # if i_episode % 10 == 0 and args.eval is True:
#     #     avg_reward = 0.
#     #     episodes = 10
#     #     for _  in range(episodes):
            
#     #         state = memory.reset_game(env.reset())
#     #         episode_reward = 0
#     #         done = False
#     #         epi_step = 0
#     #         while not done:
#     #             action = agent.select_action(state, evaluate=True)

#     #             next_state, reward, done, _ = env.step(action)
#     #             episode_reward += reward
#     #             epi_step += 1
#     #             if done:
#     #                 print(epi_step)
#     #             if epi_step== env._max_episode_steps:
#     #                 break

#     #             state = next_state
#     #         avg_reward += episode_reward
#     #     avg_reward /= episodes


#     #     writer.add_scalar('avg_reward/test', avg_reward, i_episode)

#         # print("----------------------------------------")
#         # print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
#         # print("----------------------------------------")
#     if i_episode % 100 == 0 : 
#         agent.save_model("ur5_robotiq_pick_and_place",suffix=args.suffix)
#         # print(agent.policy.weight.data)
#         for asd in zip(agent.policy.parameters()):
#             print(asd)