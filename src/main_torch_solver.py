import os
import time
from helpers import console_clear
import gym
import gym_cube
import numpy as np
from collections import namedtuple

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim

HIDDEN_SIZE = 144

COMMENT="-cubev0_remaster"
PATH = "model"+COMMENT+".pth"

class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)

class NormalizeWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(NormalizeWrapper,self).__init__(env)
    #assert isinstance(env.observation_space, gym.spaces.Discrete)
    #self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (env.observation_space.n,6), dtype=np.float32)
    return

  def observation(self, observation):
    return (observation / 5)


def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1

  #use the same environment as in training!
  env = NormalizeWrapper(gym.make("cube-v0"))
  #env = gym.make("cube-v0")
  obs_size = env.observation_space.n
  n_actions = env.action_space.n

  #Init Modell and Optimizer
  model = Net(obs_size, HIDDEN_SIZE, n_actions)
  optimizer = optim.Adam(params=model.parameters(), lr=0.001)


  #load last checkpoint
  print("Load Model: \" %s \"  "%PATH)
  checkpoint = torch.load(PATH)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  print("Loaded Model %s: "%PATH)
  print(" epoch = ", epoch)
  print(" loss = ", loss)
  # Print model's state_dict
  print("Model's state_dict:")
  for param_tensor in model.state_dict():
    print(" ", param_tensor, "\t", model.state_dict()[param_tensor].size())
    
  #set to evaluation mode
  model.eval()
  
  
  cube_scramble_count = 4 
  
  rewards = []
  steps = []
  EPISODE_CNT = 3000

  for episode in range(EPISODE_CNT):
    episode_steps = []
    obs = env.reset(scramble_count = cube_scramble_count)
    sm = nn.Softmax(dim=1)
    
    while True:
      obs_v = torch.FloatTensor([obs])
      act_probs_v = sm(model(obs_v))
      act_probs = act_probs_v.data.numpy()[0]
      
      #in training we introduced randomization on sleection of actions
      #action = np.random.choice(len(act_probs), p=act_probs)
      
      #in solver we select action with highest proability choose action based on propabilities
      max_prob = np.amax(act_probs)
      result = np.where(act_probs == max_prob)[0]
      action = result[0]

      next_obs, reward, is_done, _ = env.step(action)
      episode_steps.append(action)
      if is_done:
        rewards.append(reward)
        steps.append(episode_steps)
        #print("Episode %03d, Reward=%d" %(episode, reward))
        break
      obs = next_obs

    if(episode % 100) == 0 and episode > 0:
      reward_mean = sum(rewards) / episode
      print("Played %06d Episodes, Reward_mean=%.6f" %(episode, reward_mean))
    
  print("Playout Finished" )
  reward_mean = sum(rewards) / EPISODE_CNT
  print(" Played %06d Episodes, Reward_mean=%.6f" %(EPISODE_CNT, reward_mean)) 
  print(" Sucessful plays: %d / %d" %(sum(rewards), EPISODE_CNT))

if __name__=="__main__":
  main()
