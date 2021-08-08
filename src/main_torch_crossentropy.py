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
BATCH_SIZE = 1000
PERCENTILE = 30
GAMMA = 0.9

COMMENT="-cubev0_remaster"
PATH = "model"+COMMENT+".pth"

cube_scramble_count = 4

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


def iterate_batches(env, model, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    #obs, _ = env.reset(scramble_count = cube_scramble_count)
    obs = env.reset(scramble_count = cube_scramble_count)
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(model(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))
        if is_done:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            episode_reward = 0.0
            episode_steps = []
            #next_obs, _ = env.reset(scramble_count = cube_scramble_count)
            next_obs = env.reset(scramble_count = cube_scramble_count)
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
    
    return res, self.scramble_action_list


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

  #env = OneHotWrapper(gym.make("cube-v0"))
  env = NormalizeWrapper(gym.make("cube-v0"))
  #env = gym.make("cube-v0")
  
  #env = gym.make("cube-extrahard-v0")
  #obs_size = env.observation_space.shape[0]
  obs_size = env.observation_space.n
  n_actions = env.action_space.n
  print(env.observation_space)


  #Init Modell and Optimizer
  model = Net(obs_size, HIDDEN_SIZE, n_actions)
  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(params=model.parameters(), lr=0.001)
  
  try:
    #resume from checkpoint
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
  
    # Print optimizer's state_dict
    #print("Optimizer's state_dict:")
    #for var_name in optimizer.state_dict():
    #  print(var_name, "\t", optimizer.state_dict()[var_name])
    
    #set to training mode
    model.train()   

  except:
    print("Error on loading \"%s\", create a new save-file" %PATH)
    model = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.001)
    epoch = 0


  #Tensorboard Visu
  writer = SummaryWriter(comment=COMMENT)
  

  #torch.save(model.state_dict(), PATH)
  #return
  
  full_batch = []
  #for epoch, batch in enumerate(iterate_batches(env, model, BATCH_SIZE)): #epoch is loaded

  for batch in iterate_batches(env, model, BATCH_SIZE):
    epoch = epoch + 1
    reward_mean = float(np.mean(list(map(lambda s: s.reward, batch))))
    full_batch, obs, acts, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
    if not full_batch:
        continue
    obs_v = torch.FloatTensor(obs)
    acts_v = torch.LongTensor(acts)
    full_batch = full_batch[-1000:]

    action_scores_v = model(obs_v)
    loss_v = objective(action_scores_v, acts_v)
    loss_v.backward()
    optimizer.step()
    print("%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d" % (
      epoch, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
    writer.add_scalar("loss", loss_v.item(), epoch)
    writer.add_scalar("reward_mean", reward_mean, epoch)
    writer.add_scalar("reward_bound", reward_bound, epoch)
    writer.flush()
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_v.item(),
            }, PATH)  

    if reward_mean > 0.6:
      print("Solved!")
      #torch.save(model.state_dict(), PATH)
      break

  writer.close()
  
if __name__=="__main__":
  main()
