import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import numpy as np

class CubeExtraHardEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.action_space = spaces.Discrete(12)
    self.observation_space = spaces.Box(0, 6, shape=(6,9), dtype=np.uint8)
    #self.observation_space = spaces.Discrete((6,9))
    self.seed()
    self.viewer = None
    self.state = None
    self.steps_beyond_done = None

  def step(self, action):
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg

    self.state = []
    
    done = False

    if not done:
      reward = 0
    elif self.steps_beyond_done is None:
      # solved
      self.steps_beyond_done = 0
      reward = 1
    else:
      if self.steps_beyond_done == 0:
        logger.warn(
          "You are calling 'step()' even though this "
          "environment has already returned done = True. You "
          "should always call 'reset()' once you receive 'done = "
          "True' -- any further steps are undefined behavior."
        )
      self.steps_beyond_done += 1
      reward = 1
      return np.array(self.state), reward, done, {}

  def reset(self):
    #self.state = self.np_random.uniform(low=0, high=6, size=(6,9,))
    self.state = np.zeros((6,9), dtype=np.uint8)
    for i in range (6):
      self.state[i] = i
    self.steps_beyond_done = None
    return np.array(self.state)

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 400  
  
  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


def main():
  env = CubeExtraHardEnv()
  
  obs = env.reset()
  obs_size = env.observation_space.shape[0]
  n_actions = env.action_space.n

  
  print("Action Space:       ", env.action_space)
  print("Num Actions:        ", n_actions)
  print("Observation space:  ", env.observation_space)
  print("Observation size:   ", obs_size)
  print("Actual Observation: ", obs)


if __name__=="__main__":
  main()