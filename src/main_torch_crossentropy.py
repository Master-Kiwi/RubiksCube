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
BATCH_SIZE = 200
PERCENTILE = 30
GAMMA = 0.9


cube_scramble_count = 2

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

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset(scramble_count = cube_scramble_count)
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset(scramble_count = cube_scramble_count)
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda s: s.reward * (GAMMA ** len(s.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for example, discounted_reward in zip(batch, disc_rewards):
        if discounted_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, example.steps))
            train_act.extend(map(lambda step: step.action, example.steps))
            elite_batch.append(example)

    return elite_batch, train_obs, train_act, reward_bound

class OneHotWrapper(gym.ObservationWrapper):
  def __init__(self, env):
    super(OneHotWrapper,self).__init__(env)
    #assert isinstance(env.observation_space, gym.spaces.Discrete)
    self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape = (env.observation_space.n,6), dtype=np.float32)

  def observation(self, observation):

    res = np.copy(self.observation_space.low)

    for idx in range(0,54,1):
      value = observation[0][idx]
      res[idx][value] = 1.0
    
    return res



  


def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1

  env = OneHotWrapper(gym.make("cube-v0"))
  #env = gym.make("cube-extrahard-v0")
  obs_size = env.observation_space.shape[0]
  #obs_size = env.observation_space.n
  n_actions = env.action_space.n
  print(env.observation_space)

  net = Net(obs_size, HIDDEN_SIZE, n_actions)
  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=net.parameters(), lr=0.001)
  writer = SummaryWriter(comment="-cubev0_tweaked")
  
  full_batch = []
  for iter_no, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
    reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
    full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
    if not full_batch:
        continue
    obs_v = torch.FloatTensor(obs)
    acts_v = torch.LongTensor(acts)
    full_batch = full_batch[-500:]

    action_scores_v = net(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()
    print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
      iter_no, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
    writer.add_scalar("loss", loss_v.item(), iter_no)
    writer.add_scalar("reward_mean", reward_mean, iter_no)
    writer.add_scalar("reward_bound", reward_bound, iter_no)
    if reward_mean > 0.8:
      print("Solved!")
      break

  writer.close()

if __name__=="__main__":
  main()
