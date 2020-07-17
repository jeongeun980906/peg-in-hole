 
import argparse
import datetime
import gym
import pybullet as p
import numpy as np
import itertools
import torch
import time

from dis_env2 import UR5_robotiq
from sqnnet import SoftQNetwork
#from prioritized_memory import Memory
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

state_size=12
action_size=7
#Hyperparameters
learning_rate = 0.0005 #0.0005
gamma         = 0.98  #0.98
batch_size    = 128
alpha=0.05
soft_tau=0.01

model1=SoftQNetwork()
model2=SoftQNetwork()
target_model1=SoftQNetwork()
target_model1.load_state_dict(model1.state_dict())
target_model2=SoftQNetwork()
target_model2.load_state_dict(model2.state_dict())

memory=Memory(100000)
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)
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
        action = [0.005,0,0]
    elif  y == 1 :
        action = [-0.005,0,0]
    elif  y == 2 :
        action = [0,0.005,0]
    elif  y == 3 :
        action = [0,-0.005,0]
    elif  y == 4 :
        action = [0,0,0.005]
    elif  y == 5 :
        action = [0,0,-0.005] 
    elif  y == 6 :
        action = [0,0,-0.03] 
    
    return action     

def append_sample( state, action, reward, next_state, done):
    target = model1(Variable(torch.FloatTensor(state))).data
    old_val = target[0][action]
    target_val1 = target_model1(Variable(torch.FloatTensor(next_state))).data
    target_val2 = target_model2(Variable(torch.FloatTensor(next_state))).data
    entropy=model1.entropy(next_state)
    temp=torch.min(target_val1[0][action],target_val2[0][action])+alpha*entropy
    if done:
        target[0][action] = reward
    else:
        target[0][action] = reward + gamma* temp
    error = abs(old_val - target[0][action])
    memory.add(error, (state, action, reward, next_state, done))

def update_target_model():
    for target_param, param in zip(target_model1.parameters(), model1.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )
    for target_param2, param2 in zip(target_model2.parameters(), model2.parameters()):
        target_param2.data.copy_(
            target_param2.data * (1.0 - soft_tau) + param2.data * soft_tau
        )

def train_model():

    #mini_batch, idxs, is_weights = memory.sample(batch_size)
    tree_idx,mini_batch= memory.sample(batch_size)
    
    mini_batch = np.array(mini_batch).transpose()

    states = np.vstack(mini_batch[0])
    actions = list(mini_batch[1])
    rewards = list(mini_batch[2])
    next_states = np.vstack(mini_batch[3])
    dones = mini_batch[4]

        # bool to binary
    dones = dones.astype(int)

        # Q function of current state
    states = torch.Tensor(states)
    states = Variable(states).float()
    pred1 = model1(states)
    pred2=model2(states)
        # one-hot encoding
    a = torch.LongTensor(actions).view(-1, 1)

    one_hot_action = torch.FloatTensor(batch_size, action_size).zero_()
    one_hot_action.scatter_(1, a, 1)

    pred1 = torch.sum(pred1.mul(Variable(one_hot_action)), dim=1)
    pred2 = torch.sum(pred2.mul(Variable(one_hot_action)), dim=1)

        # Q function of next state
    next_states = torch.Tensor(next_states)
    next_states = Variable(next_states).float()
    next_pred1 = target_model1(next_states).data
    next_pred2 = target_model2(next_states).data
    
    next_pred1 = torch.sum(next_pred1.mul(Variable(one_hot_action)), dim=1)
    next_pred2 = torch.sum(next_pred2.mul(Variable(one_hot_action)), dim=1)

    entropy=model1.entropy(next_states)
    
    next_pred=torch.min(next_pred1,next_pred2)+alpha*entropy   
    rewards = torch.FloatTensor(rewards)
    dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
    target = rewards + (1 - dones) * gamma * next_pred[0]
    target = Variable(target)

    errors = torch.abs(pred1 - target).data.numpy()
        # update priority
    #for i in range(batch_size):
    #    idx = idxs[i]
    #    memory.update(idx, errors[i])
    print(tree_idx,errors)
    memory.batch_update(tree_idx,errors)

    optimizer1.zero_grad()
    #loss1 = (torch.FloatTensor(is_weights) * F.mse_loss(pred1, target)).mean()
    loss1 = (F.mse_loss(pred1, target)).mean()
    loss1.backward()
    optimizer1.step()

    optimizer2.zero_grad()
    #loss2 = (torch.FloatTensor(is_weights) * F.mse_loss(pred2, target)).mean()
    loss2 = (F.mse_loss(pred2, target)).mean()
    loss2.backward()
    optimizer2.step()

def main():
    save_action = []
    flag = 0
    flag2 = 0
    print_interval = 20 
    score = 0.0  
    last = []
    F = []
    epi=[]
    global_step=0
    for epi_n in range(500):
        state = env.reset()
        done = False
        step = 0
        score = 0
        
        #f epi_n > 2000 and epi_n%20==0:
        #    torch.save(model1,"./saved_model/model_policy_"+str(epi_n)+".pth")
        
        while not done:
            step += 1
            global_step+=1
            #print(state)
            action = model1.get_action(torch.FloatTensor(state))   #torch.from_numpy(s) > s  
            next_state, reward, done, _= env.step(action_select(action))
            state = np.reshape(state, [1, state_size])
            next_state = np.reshape(next_state, [1, state_size])
            #append_sample(state, action, reward, next_state, done)
            memory.store((state,action,reward,next_state,done))
            state = next_state
            score += reward

            update_target_model()
            
            if global_step >= 150:
                train_model()
            
            if step>300:
                done=True

            if done:
                flag += 1
                F.append(score)
                epi.append(epi_n)
                print('n_episode: ',epi_n,'score: ',score,'step: ',step)
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