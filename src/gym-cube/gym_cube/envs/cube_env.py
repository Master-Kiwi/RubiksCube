#CHANGELOG: cube_env.py
#AUTHOR: SL

#TODO: 
# documentation
# testing

#26.08.2020
#   first version

import gym
from gym import error, spaces, utils, logger
from gym.utils import seeding

import os
import time
import numpy as np
import RubiksCube as rcube
from copy import deepcopy
from helpers import console_clear

class CubeEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, metric = "quarter-turn"):
    #self.observation_space = spaces.Box(0, 6, shape=(6,9), dtype=np.uint8)
    self.seed()
    self.viewer = None
    self.state = None
    self.steps_beyond_done = None
    self.steps = None
    self.action_list = None
    self.scramble_action_list = None
    self.cube = None
    self.orig_cube = None
    self.rcube_list = None
    self.viewer = None


    
    self.metric = metric
    if(metric == "quarter-turn"):
      self.gods_no = 26
      self.action_space = spaces.Discrete(12)
    elif(metric == "half-turn"): 
      self.gods_no = 20
      self.action_space = spaces.Discrete(18)   # U2, D2, R2, L2, F2, B2
    else:
      assert True, "Metric not supported"
      exit(0)

    #2D model (6*3*3 faces)
    self.observation_space = spaces.Discrete(54)
    #self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (54,6), dtype=np.float32)
    #self.observation_space = spaces.Box(low = 0.0, high = 5.0, shape = (54, 1), dtype = np.float32)


  def step(self, action):
    err_msg = "%r (%s) invalid" % (action, type(action))
    assert self.action_space.contains(action), err_msg
    self.steps += 1



    #carry out action
    self.action_list.append(action)
    if(action >= 12):     #half-turn action (2x quarter-turn action)
      self.cube.actions_simple(action - 12)
      self.cube.actions_simple(action - 12)
    else:                 #quarter turn action
      self.cube.actions_simple(action)

    self.state = self.cube.get_state(flatten = True)
    
    visited = False
    #check all visited states before
    for item in self.rcube_list:
      visited = self.cube.equals(item)    
      if visited is True: 
        break          #state already visited

    


    result = bool(self.cube.equals(self.orig_cube) )
    #done = False
    done = bool( 
      result                              #puzzle solved
      or self.steps > self.gods_no        #gods no reached
      or visited                          #state already seen
      )

    if result is True:      reward = 1
    else:                   reward = 0

    self.rcube_list.append(deepcopy(self.cube))          #store alle states
    return np.array(self.state), reward, done, self.action_list

  #
  def reset(self, scramble_count=None, scramble_list=None):

    self.state = np.zeros(54, dtype=np.uint8)
    #self.state = np.zeros(shape=(54,6),dtype=np.uint8) 
    #self.state = self.np_random.uniform(low=0, high=6, size=(6,9,))

    self.steps_beyond_done = None
    self.steps = 0
    self.action_list = []
    self.scramble_action_list = []
    self.cube =       rcube.tRubikCube()         #work is carried on in this object
    self.orig_cube =  rcube.tRubikCube()         #remainder for start condition
    self.rcube_list = []                               #remainder for all steps taken


    if scramble_count is not None:
      err_msg = "initial scramble incorrect: %d " % (scramble_count)
      assert scramble_count <= self.gods_no, err_msg
      assert scramble_count > 0, err_msg
      actual_cube = rcube.tRubikCube()    
      rcube_list = []
      rcube_list.append(actual_cube)

      while scramble_count > 0:
        test_cube = deepcopy(actual_cube)
        #execute random action
        random_action = self.action_space.sample()

        if(random_action >= 12):     #half-turn action (2x quarter-turn action)
          test_cube.actions_simple(random_action - 12)
          test_cube.actions_simple(random_action - 12)
        else:                 #quarter turn action
          test_cube.actions_simple(random_action)

        equal = False
        #check all visited states before
        for item in rcube_list:
          equal = test_cube.equals(item)    
          if equal is True: break          #state already visited -> drop it
        if equal is True: 
          continue          #state already visited -> drop it

        #new state > add to list
        actual_cube = deepcopy(test_cube)
        rcube_list.append(actual_cube)
        self.scramble_action_list.append(random_action)
        scramble_count -= 1
      #set internal state to scrambled state
      self.cube = deepcopy(actual_cube)
    
    if scramble_list is not None:
      for action in scramble_list:
        if(action >= 12):     #half-turn action (2x quarter-turn action)
          self.cube .actions_simple(action - 12)
          self.cube .actions_simple(action - 12)
        else:                 #quarter turn action
          self.cube .actions_simple(action)
        self.scramble_action_list.append(action)


    self.cube.clear_action_list()
    #cube data is flat 1D array size 54, color values ranging from 0 to 6
    observation = self.cube.get_state(flatten = True)

    #one hot encoding
    #for idx in range(0,len(observation),1):
      #value = observation[idx]
      #self.state[idx][value] = 1.0

    self.state = observation

    return np.array(self.state), self.scramble_action_list

  def render(self, mode='human'):
    screen_width = 600
    screen_height = 400 
    cubewidth = 20
    cubeheight = 20

    col = [None] * 7
    col[rcube.COL_IDX_WHITE]  = [1,1,1]
    col[rcube.COL_IDX_YELLOW] = [1,1,0]
    col[rcube.COL_IDX_ORANGE] = [1,0.5,0]
    col[rcube.COL_IDX_RED]    = [1,0,0]
    col[rcube.COL_IDX_GREEN]  = [0,1,0]
    col[rcube.COL_IDX_BLUE]   = [0,0,1]
    col[rcube.COL_IDX_BLACK]  = [0,0,0]
    
    #init objects and rendering here
    if self.viewer is None:
      from gym.envs.classic_control import rendering
      self.viewer = rendering.Viewer(screen_width, screen_height)
      
      #background is defaulted to white, first draw a filled rect to clear
      backg = rendering.FilledPolygon([(0, 0), (0, screen_height), (screen_width, screen_height), (screen_width, 0)])        
      backg.set_color(0,0,0)  
      self.viewer.add_geom(backg)
      
      #x/y positions - for each 3x3 side, 0/0 ist bottom left
      xpos_1 = 20
      xpos_2 = xpos_1 + (cubewidth + cubewidth/2) * 3.5
      xpos_3 = xpos_2 + (cubewidth + cubewidth/2) * 3.5
      xpos_4 = xpos_3 + (cubewidth + cubewidth/2) * 3.5
      ypos_1 = screen_height-20
      ypos_2 = ypos_1 - (cubeheight + (cubeheight/2)) * 3.5
      ypos_3 = ypos_2 - (cubeheight + (cubeheight/2)) * 3.5

      for idx, col_idx in enumerate(self.state, start = 0):
        #Back = RED     
        if idx == 27:
          ypos = ypos_1
          xpos = xpos_2
        if idx == 30: 
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2
        if idx == 33: 
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2

        #left = Blue
        if idx == 36:
          ypos = ypos_2
          xpos = xpos_1
        if idx == 39:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_1
        if idx == 42:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_1

        #up = white
        if idx == 0:
          ypos = ypos_2
          xpos = xpos_2
        if idx == 3: 
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2
        if idx == 6: 
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2

        #right = green
        if idx == 45:
          ypos = ypos_2
          xpos = xpos_3
        if idx == 48:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_3
        if idx == 51:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_3

        #down = yellow
        if idx == 9:
          ypos = ypos_2
          xpos = xpos_4
        if idx == 12:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_4
        if idx == 15:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_4

        #front = orange
        if idx == 18:
          ypos = ypos_3
          xpos = xpos_2
        if idx == 21:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2
        if idx == 24:
          ypos -= cubeheight + (cubeheight/2)
          xpos = xpos_2

        #initial object definition (position and color), square is drawn "centered"
        l, r, t, b = (-cubewidth / 2), (cubewidth / 2), (cubeheight / 2), (-cubeheight / 2)
        rendercube = rendering.FilledPolygon([(l+xpos, b+ypos), (l+xpos, t+ypos), (r+xpos, t+ypos), (r+xpos, b+ypos)])
        rendercube.set_color(col[col_idx][0], col[col_idx][1], col[col_idx][2])
        self.viewer.add_geom(rendercube)

        #x distance between squares 1.5x width
        xpos += cubewidth + (cubewidth / 2) 

    if self.state is None:
      return None  
   

    #called on each "render" - change the color for the geom objects
    for idx, geom in enumerate(self.viewer.geoms, start = -1):
      if idx == -1: continue;         #background object
      col_idx = self.state[idx]       #color index for each surface
      
      #"col" > mapping of col_idx to r,g,b
      geom.set_color(col[col_idx][0], col[col_idx][1], col[col_idx][2])  #set_color(r,g,b)
    
    
    #this call renders the picture to screen
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')  


  #random seed
  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]  

  def close(self):
    if self.viewer:
      self.viewer.close()
      self.viewer = None

def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1

  #env = CubeEnv(metric="half-turn")
  env = CubeEnv(metric="quarter-turn")
  #env_test(env)
  #random_agent(env)
  render_test(env)
  
  exit(0)

def render_test(env):
  print("------Reset Environment to Orig------------" )
  #obs, scramble_actions = env.reset()
  
  
  while True:
    obs, scramble_actions = env.reset()
    env.render(mode="human")
    time.sleep(1)

    actions_list = [0,1,2,3,4,5,6,7,8,9]  
    for step, action in enumerate(actions_list, start = 1):
      obs, reward, done, info = env.step(action)
      print("step: %02d, Done: %s, Reward: %d , obs: %s" % (step, done, reward, obs))
      print("Actual Observation: ", obs)    
      env.render(mode = "human")
      time.sleep(0.1)

  input("Press Enter to Exit")
  env.close()




def env_test(env):
  scramble_cnt = 2
  print("------Reset Environment to Orig------------" )
  obs, scramble_actions = env.reset()
  #obs_size = env.observation_space.shape[0]
  n_actions = env.action_space.n
  #obs_size = len(env.observation_space)
  
  print("Action Space:       ", env.action_space)
  print("Num Actions:        ", n_actions)
  print("Observation space:  ", env.observation_space)
  #print("Observation size:   ", obs_size)
  print("Actual Observation: ", obs)
  print("scramble_actions:   ", scramble_actions)
  

  #sample random action
  random_action = env.action_space.sample()
  obs, reward, done, info = env.step(random_action)
  print("\n")
  print("Perform random act: ", random_action)
  print("Reward/Done: %d; %s " % (reward, done))
  print("Actual Observation: ", obs)
  print("Env Info:           ", info)
  exit(0)

  #quarter turn metric action has counter action, half turn metric action is just repeated
  if random_action < 12:
    action = env.cube.conj_action(random_action)
  else:
    action = random_action

  obs, reward, done, info = env.step(action)
  print("\n")
  print("Perform reverse act: ", action)
  print("Reward/Done: %d; %s " % (reward, done))
  print("Actual Observation: ", obs)
  print("Env Info:           ", info)


  #Reset env and bring it back to normal
  scramble_cnt = env.gods_no
  obs, scramble_actions = env.reset(scramble_count=scramble_cnt)
  print("\n --- RESET ENVIRONMENT ---- ")
  print("Actual Observation: ", obs)
  print("scramble_actions:   ", scramble_actions)
  counter_actions_list = []
  for action in scramble_actions:
    counter_actions_list.append(env.cube.conj_action(action))  
  counter_actions_list.reverse()
  print("Perform reverse sequence: ", counter_actions_list)

  for step, action in enumerate(counter_actions_list, start = 1):
    obs, reward, done, info = env.step(action)
    print("step: %02d, Done: %s, Reward: %d , obs: %s" % (step, done, reward, obs))
  print("Actual Observation: ", obs)
  

def random_agent(env):

  #random_actions = [0,4,8,15,3,1,7,12,1]
  random_actions = [0,4,8,3]
  obs, scramble_actions = env.reset(scramble_list = random_actions)
  counter_actions_list = []
  for action in scramble_actions:
    counter_actions_list.append(env.cube.conj_action(action))  
  counter_actions_list.reverse()
  

  env_initial = deepcopy(env)
  episode_cnt = 0
  episodes_reward = []
  
  while True:
    env.reset(scramble_list = random_actions)
    
    while True:
      random_action = env.action_space.sample()
      obs, reward, done, info = env.step(random_action)
      if done is True: 
        break
    episodes_reward.append(reward)    
    episode_cnt += 1
    if(reward == 1): 
      break
    if(episode_cnt % 100 == 0):
      console_clear()
      print("Played %08d random Episodes, no solution" % episode_cnt)
      env_initial.cube.print_2d()
      env.cube.print_2d()
    
  if(reward == 1):
    console_clear()
    print("Solution found in %08d Episodes" %episode_cnt)   
    print("Scramble Actions:      ", scramble_actions)
    print("Sequence to solve:     ", info)

    env_initial.cube.print_2d()
    env.cube.print_2d()


  


if __name__=="__main__":
  main()