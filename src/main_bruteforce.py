#CHANGELOG: main.py
#AUTHOR: SL

#04.01.2020 
#introduced scoring, requires RubiksCube.py from 04.01.2020
#  runs iteration and determines also the score-value for each iteration
#  displays the max-score results per depth 
#  if max_score > overall_best_score --> save test_cube to file
#  this example shows on real_cube.json, that following the way of_max_score will not find a solution with limited search depth (6)
#  for example when all is correct instead of corners then it needs more then 6 actions to yield a better score then initial.
#  if you follow the max-score path then you will yield action lists that typical show conjugate pairs of actions that end in always the same cube state
#   for example: [11, 10, 4, 5]  [11, 10, 5, 4]  [11, 11, 5, 5]  (all values with difference of 6 are conjugates)
#  the actions that would bring the solution have less score because the corner-swap sequence is not complete and the cube is more "incorrect" from score()
#  with limitation to 6 actions it is possible to finish a swap/rotata of edges sequence, so as long as edges are incorrect we can fix this we consecutive runs at depth=6
#  think about the definition of score() 

#28.12.2019:
#improved performance:
#  introduced list comprehension
#  deepcopy() is slow - so instead of test_cube = deepcopy(Cube) we start with test_cube = original-cube 
#  instead of comparing to orginal cube we compare to wanted cube
#  when solution is found then test_cube.actions_list[] needs to be conjugated and reversed, to get the actions that need to be done.

#27.12.2019
#-first version
#  simple example in showing how to brute-force search a solution with iterate_tools.py
#  load som pre-rotated file to tRubikCube Instance
#  try to find a solution by increasing search depth and running all iterations for the "test-cube"
#  solution means to find a match with the original-cube
#  then test_cube.actions_list[] contains the steps that has to be done to solve the test-cube

import os 
import numpy as np
import datetime
import time
import random
from copy import copy, deepcopy

#self written submodules
from RubiksCube import tRubikCube
from iterate_tools import tIterate
from helpers import console_clear



def main():
    #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1
  Cube = tRubikCube()
  Orig_Cube = tRubikCube()
  #Orig_Cube.save_to_file("orig_cube.json")

  #random rotate a cube starting from original state 
  random.seed()
  num_rotations = 12
  for i in range (num_rotations):
    action=random.randrange(0, Cube.num_actions())
    Cube.actions(action)
  #Cube.save_to_file("simple_cube_04.json")
  #Cube.save_to_file("moderate_cube_08.json")
  #Cube.save_to_file("complex_cube_12.json")


  #------load a prerotated cube-----------
  #filename = "../data/simple_cube_04.json"
  #filename = "../data/moderate_cube_08.json"
  #filename = "../data/complex_cube_12.json"
  filename = "../data/real_cube.json"

  script_dir        = os.path.dirname(__file__) #<-- absolute dir the script is in
  cube_file_path    = os.path.join(script_dir, filename)
  Cube.load_from_file(cube_file_path)  
  

  #Draw Starting Cube it and display some information
  Cube.print_2d()
  result = Cube.self_test()
  print("Cube Self Test Result: %s" % str(result))
  if result == False: exit(0)
  starting_score = Cube.score()
  print("Cube Starting score: %02d" % starting_score)

  num_rotations = len(Cube.actions_list)
  #no solution in cube.file, try with 12 
  if num_rotations == 0: 
    num_rotations = 7
    test_solution= []   
    print("No Sequence in Cube-Data found")
  else:
    print("Action sequence:                     " + str(Cube.actions_list))
    #calculate the solution sequence that should be found
    test_solution = Cube.get_conj_action_list()
    test_solution.reverse()
    print("Conjugate action sequence (solution):" + str(test_solution))
  
  num_rotations = 7  
  print("Search Solution with Iterative Deepening, depth=%d\n" % num_rotations)
  print("Press Enter to continue")
  input()
  console_clear()

  #solutions = []
  Solved_Cubes = []
  iter_per_sec = 2000 #approx
  overall_best_score = starting_score

  #increment the depth iterative up to num_rotations
  for itv_deep in range(1, num_rotations+1):
    if Solved_Cubes: break    #solution found on last depth --> break
    seq_iterator = tIterate(itv_deep, 12, 0)
    iter_steps   = seq_iterator.get_total_num()
    iter_func    = seq_iterator.generator()
    
    time_to_completion = float(iter_steps / iter_per_sec)
    print("\nIteration Depth:  %d" % itv_deep, end='')
    if iter_steps > 10000:
      print("  Num_Iterations:   %dk" % (iter_steps/1000), end='')
    else:
      print("  Num_Iterations:   %d" % iter_steps, end='')
    
    if(time_to_completion > 60): 
      print("  Estimated Time:   %.2fmin" % (time_to_completion/60.0) )
    else:
      print("  Estimated Time:   %.2fsec" % time_to_completion )
    #print("  Initial Iter:   "+str(Orig_Cube.next_iteration()))
    
    start = datetime.datetime.now()
    iter_last = 0
    max_score = -1
    #max_score = starting_score
    score_list = []
    #score_list = [0] * 2
    skipped_iter_steps = 0

    #try all sequences and compare with original cube
    for iter in range(iter_steps):
      #iteration_step = seq_iterator.next()      #get next sequence
      iteration_step = next(iter_func).copy()           #using generator is faster
      iteration_step.reverse()
 
      #check the action list for unnecessary sequences
      #as we brute-force all combinations we can drop:
      #  each sequence containing at least 1 action/counteraction pair as they negate each other
      #  each sequence containing 3x same action 3 x rotate(CW) = 1 x rotate(CCW)
      last_action = iteration_step[0]
      action_counter = 1
      skip_step = False
      for i in range(1, len(iteration_step)):
        action = iteration_step[i]
        if action == Cube.conj_action(last_action):
          skip_step = True
          break
        
        if action == last_action:
          action_counter += 1        
          if action_counter == 3:
            skip_step = True
            break
        else: 
          action_counter = 1
        last_action = action
      
      #no need to compute this sequence
      if skip_step:
        skipped_iter_steps += 1
        continue

      #we canot use "=" as this creates a reference, each .actions call modifies the cube data
      #create a copy of the modified cube. alternative we could create a new cube (original) and try to find the solution to come to the modified state.
      #finally the 2nd solution is faster then deppcopy()
      Test = tRubikCube()
      #Test = deepcopy(Cube)  
      
      #Perform all rotate actions, iteration_step is a list of actions, list comprehension is faster
      [Test.actions_simple(action) for action in iteration_step]
      #for action in iteration_step:
        #Test.actions(action)    #rotate the cube 
      #check for match with target, we need to do this only after the rotation is complete, as we checked all other variants before
      score = Test.compare(Cube)
      if score >= max_score: 
        max_score=score
        new_entry = [score, iteration_step.copy()]
        score_list.append(new_entry)
        


      if Test.equals(Cube): 
        print("\n----!!!Solution found: %s!!!-----" % str(iteration_step))
        Solved_Cubes.append(deepcopy(Test))
      #subtraction is faster then modulo operation, adapt to CPU-speed or use modulo, will be true approx each second, will not ouput on fast runs, module will fire on start
      #combine modulo and estimated iter_per_sec, this will be true approx each second or each % of completion
      itercnt = iter-iter_last
      #if (itercnt >= iter_per_sec):  
      #if (itercnt >= iter_per_sec) or (iter % (int(iter_steps/100)+1) == 0):  
      if (itercnt >= iter_per_sec) or not (iter % (int(iter_steps/100)+1)):  
        iter_last = iter
        actual = datetime.datetime.now()
        actual_delta = actual-start
        actual_delta = actual_delta.total_seconds() 
        if(actual_delta > 0.0) and (iter > 0):
          iter_per_sec = iter / actual_delta
          progress = 100.0*iter/iter_steps
          remaining = iter_steps-iter
          time_to_completion = remaining / iter_per_sec
          if time_to_completion > 60:
            print("\r[Progress=%.3f%%  iter_num=%dk/%dk  iter_per_sec=%05d  remaining=%.2fmin  skipped=%d]          " % (progress, (iter/1000), (iter_steps/1000), iter_per_sec ,(time_to_completion/60), skipped_iter_steps ), end='')
          else:
            print("\r[Progress=%.3f%%  iter_num=%dk/%dk  iter_per_sec=%05d  remaining=%.2fsec  skipped=%d]          " % (progress, (iter/1000), (iter_steps/1000), iter_per_sec ,time_to_completion, skipped_iter_steps ), end='')



    print("\n-----------Statistics for this run----------")
    #Output Statistics for this run
    stop = datetime.datetime.now()
    delta = stop-start
    delta_seconds = float(delta.total_seconds())
    if delta_seconds > 60:
      print(" Total time = %.2fmin" % (delta_seconds/60.0))
    else:
      print(" Total time = %.2fsec" % delta_seconds)

    #iter_per_sec estimation should be correct as this is needed for runtime estimation upon start
    if delta_seconds > 0.0: 
      iter_per_sec = iter_steps / delta_seconds
      print(" Iterations per seconds = %d" % int(iter_per_sec))
    else:
      print(" Iterations per seconds = <not reliable, to short>")

    if iter_steps > 10000:
      print("  Skipped Iteration Steps = %d/%dk" % (skipped_iter_steps, iter_steps))
    else:
      print("  Skipped Iteration Steps = %d/%d" % (skipped_iter_steps, iter_steps))
    
    
    print("Starting Cube State - Score: %02d" %starting_score)
    Cube.print_2d()
    #print("Cube Score-List: %s" % str(score_list))
    if score_list:
      max_score = max(score[0] for score in score_list)
    
    best_sequences = []
    for score in score_list:
      if score[0] == max_score:
        best_sequences.append(score[1].copy())
    print("Iteration Max Score: %02d   Number of Sequences @ max_score: %02d" % (max_score, len(best_sequences)))
    print("Overall Best Score:  %02d" % overall_best_score)

    print("best action sequences: %s" % str(best_sequences))
    
    sequence_cnt = 0
    #output of best score sequences
    for sequence in best_sequences:
      #first we need to revert and conjugate the action list that was found 
      temp_cube = tRubikCube()
      for actions in sequence:
        temp_cube.actions_simple(actions)
      solved_sequence = temp_cube.get_conj_action_list()
      solved_sequence.reverse()
      
      #now we can back rotate the original given cube
      score_visu_cube = deepcopy(Cube)
      #empty it's action list - we are just interested in all new actions
      #score_visu_cube.clear_action_list()
      #print("  Actual Visualization for: %s" % (solved_sequence))
      for actions in solved_sequence:
        score_visu_cube.actions_simple(actions)
        #print("  [%d]:%s" % (actions, Cube.action_dict.get(actions)))
      #score_visu_cube.print_2d()
      
      #our sequences are better then the best known sequence -> save them to file
      if max_score > overall_best_score:
         max_score_filename = "%s_score_%02d_%02d.json" %(filename, max_score, sequence_cnt)
         print("  max_score > overall_best_score -> save this cube to file: %s" % max_score_filename) 
         score_cube_file_path    = os.path.join(script_dir, max_score_filename)
         score_visu_cube.save_to_file(score_cube_file_path)
         sequence_cnt += 1
    
      #print("Press enter for next sequence")
      #time.sleep(0.5)
      #input()

    #all sequences shown --> set overall_best_score if a solution was better
    if max_score > overall_best_score:
       overall_best_score = max_score
          
    #no solution on actual depth --> next depth
    if not Solved_Cubes:
      print(" No Solution found")
    
    print("Press enter for next Iteration depth")
    input()
    console_clear()
    
  #we found solutions - output all of them
  if Solved_Cubes:
    print("\n-------------SOLUTIONS FOUND----------------")
    if test_solution:
      print("Known solution from beginning        :%s" % str(test_solution))
    else: 
      print("No Known solution from beginning")
    i = 0
    for Cube in Solved_Cubes:
      #print("  Action sequence found              :" + str(Cube.actions_list))
      #calculate the solution sequence that should be found
      test_solution = Cube.get_conj_action_list()
      test_solution.reverse()
      print("  Solution sequence found [%02d]       :%s" % (i, str(test_solution)))
      i += 1
  

if __name__=="__main__":
  main()

exit(0)