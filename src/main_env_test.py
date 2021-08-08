import os
import time
from helpers import console_clear
import gym
import gym_cube


def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1

  env = gym.make("cube-v0")
  #env = gym.make("cube-extrahard-v0")
  
  obs_size = env.observation_space.n
  n_actions = env.action_space.n


  #environment Info
  print("Action Space:       ", env.action_space)
  print("Num Actions:        ", n_actions)
  print("Observation space:  ", env.observation_space)
  print("Observation size:   ", obs_size)


  #environment at solved state
  obs = env.reset()                                 
  env.render(mode = "human")
  print("---Env in Reset State = Solved---")
  print("Actual Observation: ", obs)
  input("Press Enter to Continue")


  #defined scramble actions
  random_actions = [0,0]
  obs = env.reset(scramble_list = random_actions)   
  env.render(mode = "human")
  print("---Env with predefined scramble actions = %s---" %random_actions)
  #print("Scramble Actions:   ", scramble_actions)
  print("Actual Observation: ", obs)
  input("Press Enter to Continue")

  #defined scramble count
  random_scramble_count = 4
  obs = env.reset(scramble_count = random_scramble_count)               
  env.render(mode = "human")
  print("---Env with %02d random scramble actions" %random_scramble_count )
  #print("Scramble Actions:   ", scramble_actions)
  print("Actual Observation: ", obs)
  input("Press Enter to Continue")

  
  
  
  #simple random agent
  print("---Simple random agent playing---" )
  random_actions = [0,1,2,3]
  obs = env.reset(scramble_list = random_actions)
  #print("Scramble Actions:   ", scramble_actions) 
  input("Press Enter to Continue")

  episode_cnt = 0
  render_once = True

  while True:
    #set to starting condition, get initial obs
    obs = env.reset(scramble_list = random_actions)

    #play 1 episode
    while True:
      random_action = env.action_space.sample()
      obs, reward, done, info = env.step(random_action)
      
      if render_once is True:
        env.render(mode = "human")
        #time.sleep(0.1)

      if done is True:  #done condition: solved, max steps, 26 or 20, obs already seen
        break
    
    episode_cnt += 1

    if render_once is True:
      render_once = False

    if(reward == 1): #will only yield 1 if solved
      break
    
    #simple progress notification on console
    if(episode_cnt % 100 == 0):
      console_clear()
      print("Random Agent Played %08d Episodes, no solution found" % episode_cnt)
      render_once = True

  print("---Environment solved from random agent---")
  print(" Played %d Episodes" %episode_cnt)
  print(" Solution found:       ", info)
  #print(" Scramble Code was:    ", scramble_actions)
  input("-Press Enter to see solution-")
  
  #Present the solution
  obs = env.reset(scramble_list = random_actions)
  env.render(mode = "human")
  
  for action in info:
    time.sleep(2)
    print(" Action: %d" %action)
    obs, reward, done, info = env.step(action)
    env.render(mode = "human")
  
  print("Rendering Complete")
  input("-Press Enter to Exit-")
  env.close()
  
  exit(1)


if __name__=="__main__":
  main()


