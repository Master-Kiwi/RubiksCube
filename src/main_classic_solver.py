#CHANGELOG: main_classic_solver.py
#AUTHOR: SL

#   TODO: 
#     code "beautfication"
#     bruteforce add's actions to the cube when it is solved upon start [1,1,1,]
#     rework step 1, without bruteforce 
#     rework step 2, without bruteforce 
#     unify output text of solver methods
#     add "help" cmd to terminal

#20.02.2020 
#  changed cmd="solve"  
#     display of solution sequence in letters instead numbers
#     grouped output of solution sequences (per step)
#  changed default terminal output 
#     Moves = len(action_list)
#     actions are in short notation text (U, U' ) instead of numbers
#  added CONSOLE_COLS and CONSOLE_LINES, changing this variables to adapt size of console window
#     suggestion is to use 120 / 60
#  using blink_line() from helpers.py, needs helpers.py from 11.02.2020

#11.02.2020 
#  initial version 
#   will solve the cube using beginners method from ruwix.com (7-steps)
#   simple terminal with save/load/new/shuffle/solve,...
#   solve it manual / test algos or solve it automatic
#   testmode for continuous solving

import os
import time
import random
import numpy as np
from copy import copy, deepcopy

#self written submodules
import RubiksCube as cube
import iterate_tools as itertools
from helpers import console_clear, num_to_str_si, sec_to_str, blink_line

CONSOLE_COLS = 120
CONSOLE_LINES = 60

def main(): 
  os.system("mode con: cols="+ str(CONSOLE_COLS) +"lines="+ str(CONSOLE_LINES))  #12*4 +1
  
  console_clear()
  Cube = cube.tRubikCube()
  orig_cube   = cube.tRubikCube()
  orig_edge   = orig_cube.get_edge(sort=True)
  orig_corner = orig_cube.get_corner(sort=True)
  
  #------load a prerotated cube-----------
  #filename = "../data/simple_cube_04.json"
  #filename = "../data/moderate_cube_08.json"
  #filename = "../data/midlayer_test.json"
  filename = "../data/real_cube_03.json"
  filename = "../data/real_cube_04.json"
  #filename = "../data/complex_cube_12.json"
  #filename = "../data/real_cube.json"

  script_dir        = os.path.dirname(__file__) #<-- absolute dir the script is in
  cube_file_path    = os.path.join(script_dir, filename)
  Cube.load_from_file(cube_file_path)  
  result = Cube.self_test()
  if (result == False):
    print("!!!!Cube Data incorrect!!!!")

  cmd = ""
  info = ""
  edge_location = ""
  corner_location = ""


  while True:
    #get edge and corner position in actual Cube
    edge_location   = Cube.search_edge(orig_edge)
    corner_location = Cube.search_corner(orig_corner)
    sequence = Cube.get_action_list(notation="short")
    moves = len(Cube.get_action_list())
    print("Last Command: %s (%s)" % (cmd, info))
    print("Edge   location: %s " % edge_location)
    print("Corner location: %s " % corner_location)
    print("Moves=%04d     : %s " % (moves, sequence))
    #print("Actual Sequence: ")
    #Cube.print_action_list(max_line_len=CONSOLE_COLS)

    Cube.print_2d()
    print(" rotate CW commands:  u/u'..upper d/d'..down r/r'..right l/l'..left f/f'..front  b/b'..back")
    print(" file commands:       reload..reload original file   save..state > progress fale  load..stat < progess file " )
    print(" resolve commands:    1...white edges  2...white corners")
    print(" quit to exit")
    cmd = str(input("cmd:>> "))
    info = ""

    if "file" == cmd:
      info = "Orig Filename: %s" % filename
      console_clear()
      continue
    
    if "setfile" == cmd:
      filename = "../data/" + str(input("Enter new Filename: ")) + ".json"
      info = "New Filename: %s" % filename
      console_clear()
      continue

    if "load" == cmd:
      cube_file_path    = os.path.join(script_dir, filename)
      Cube.load_from_file(cube_file_path) 
      info = "Cube state reloaded from original file %s" % filename
      console_clear()
      continue
    
    if "save" == cmd:
      cube_file_path    = os.path.join(script_dir, filename)
      Cube.save_to_file(cube_file_path)  
      inf = "Save state to Original file %s" % filename
      console_clear()
      continue

    if "qload" == cmd:
      progress_filename = filename+"_quickload.json"
      cube_file_path    = os.path.join(script_dir, progress_filename)
      Cube.load_from_file(cube_file_path)  
      info = "Cube state loaded from progress file %s" % progress_filename
      console_clear()
      continue

    if "qsave" == cmd:
      progress_filename = filename+"_quicksave.json"
      cube_file_path    = os.path.join(script_dir, progress_filename)
      Cube.save_to_file(cube_file_path)  
      inf = "Save state to progress file %s" % progress_filename
      console_clear()
      continue
    
    if "new" == cmd:
      Cube = cube.tRubikCube()
      info="Original Cube"
      console_clear()
      continue
    
    if "shuffle" == cmd:
      random.seed()
      num_rotations = 500
      for i in range (num_rotations):
        action=random.randrange(0, Cube.num_actions())
        Cube.actions_simple(action)      
      
      Cube.clear_action_list()      
      console_clear()
      continue
    if "quit" == cmd:
      exit(0)

    if "clear" == cmd:
      info = "Clear the actions list"
      Cube.clear_action_list()
      console_clear()
      continue
    
    if "1" == cmd:
      print("Bruteforce solve white edges (white-cross)...")
      search_depth = 6
      Cube.clear_action_list()
      Cube=bruteforce(Cube,search_depth,11)
      Cube=bruteforce(Cube,search_depth,12)
      Cube=bruteforce(Cube,search_depth,13)
      Cube=bruteforce(Cube,search_depth,14)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue
    
    if "2" == cmd:
      print("Bruteforce solve white side (white-corners)...")
      search_depth = 6
      Cube.clear_action_list()
      Cube=bruteforce(Cube,search_depth,21)
      Cube=bruteforce(Cube,search_depth,22)
      Cube=bruteforce(Cube,search_depth,23)
      Cube=bruteforce(Cube,search_depth,24)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue
    
    if "3" == cmd:
      Cube.clear_action_list()
      #solve each edge
      for i in range(4,8,1):
        Cube=solve_second_layer_edge(Cube, i)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)      
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue

    if "4" == cmd:
      Cube.clear_action_list()
      Cube=solve_yellow_cross(Cube)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue

    if "5" == cmd:
      Cube.clear_action_list()
      Cube = solve_yellow_edge(Cube)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue
    
    if "6" == cmd:
      Cube.clear_action_list()
      Cube = position_yellow_corner(Cube)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue

    if "7" == cmd:
      Cube.clear_action_list()
      Cube = orient_last_layer_corner(Cube)
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      Cube.print_2d()
      print("Press Enter to continue")
      input()
      console_clear()
      continue

     

    if "solve" == cmd:
      Cube.clear_action_list()
      complete_action_list = []
      
      print("1. Bruteforce solve white edges (white-cross)...")
      Cube.clear_action_list()
      search_depth = 6
      Cube=bruteforce(Cube,search_depth,11)
      Cube=bruteforce(Cube,search_depth,12)
      Cube=bruteforce(Cube,search_depth,13)
      Cube=bruteforce(Cube,search_depth,14)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      print("2. Bruteforce solve white side (white-corners)...")
      search_depth = 6
      Cube.clear_action_list()
      Cube=bruteforce(Cube,search_depth,21)
      Cube=bruteforce(Cube,search_depth,22)
      Cube=bruteforce(Cube,search_depth,23)
      Cube=bruteforce(Cube,search_depth,24)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()


      print("3. Solve Second layer...")
      Cube.clear_action_list()
      for i in range(4,8,1):
        Cube=solve_second_layer_edge(Cube, i)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      print("4. Solve yellow cross...")
      Cube.clear_action_list()
      Cube=solve_yellow_cross(Cube)  
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      print("5. Solve yellow edge...")
      Cube.clear_action_list()
      Cube = solve_yellow_edge(Cube)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      print("6. Position yellow corner...")
      Cube.clear_action_list()
      Cube = position_yellow_corner(Cube)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      print("7. Orient Last layer yellow corner...")
      Cube.clear_action_list()
      Cube = orient_last_layer_corner(Cube)
      moves = len(Cube.get_action_list())
      print("%d Moves:" % moves)    
      Cube.print_action_list(max_line_len=CONSOLE_COLS)
      complete_action_list += Cube.get_action_list()

      Cube.clear_action_list()
      Cube.actions_list = complete_action_list

      #Save Solution to File
      solution_filename = filename[:-5] + "_solved.json"
      solution_cube_file_path    = os.path.join(script_dir, solution_filename)
      print("Save Solution to File: {}" .format(solution_filename))
      Cube.save_to_file(solution_cube_file_path)  

      moves = len(Cube.get_action_list())
      #Cube.print_2d()
      blink_line("Defeated Rubik's Cube in {} moves" .format(str(moves)), 5, 0.25)
      input()
      console_clear()
      continue

    if "test" == cmd:
      cubes_solved = 0
      while True:
        console_clear()
        print("Cube Test Mode Running....")
        print(" Cube's Solved: %d" %cubes_solved)
        cubes_solved += 1

        random.seed()
        num_rotations = 500
        for i in range (num_rotations):
          action=random.randrange(0, Cube.num_actions())
          Cube.actions_simple(action)      
        Cube.clear_action_list()    
        Cube.print_2d()
        print("Start Solving...")
        search_depth = 6
        Cube=bruteforce(Cube,search_depth,11)
        Cube=bruteforce(Cube,search_depth,12)
        Cube=bruteforce(Cube,search_depth,13)
        Cube=bruteforce(Cube,search_depth,14)
        Cube=bruteforce(Cube,search_depth,21)
        Cube=bruteforce(Cube,search_depth,22)
        Cube=bruteforce(Cube,search_depth,23)
        Cube=bruteforce(Cube,search_depth,24)
        for i in range(4,8,1):
          Cube = solve_second_layer_edge(Cube, i)
        Cube = solve_yellow_cross(Cube)  
        Cube = solve_yellow_edge(Cube)
        Cube = position_yellow_corner(Cube)
        Cube = orient_last_layer_corner(Cube)
        Cube.print_2d()
        Cube.print_action_list(max_line_len=CONSOLE_COLS)

        moves = len(Cube.get_action_list())
        blink_line("Defeated Rubik's Cube in {} moves" .format(str(moves)), 5, 0.25)
        Cube.clear_action_list()
      


    if "31" == cmd:
      Cube.swap_second_layer_edge(cube.SIDE_IDX_RIGHT,"right",1)  #right edge on right side = edge #0 +4
    if "32" == cmd:
      Cube.swap_second_layer_edge(cube.SIDE_IDX_FRONT,"right",1)  #right edge on front side = edge #1 +4
    if "33" == cmd:
      Cube.swap_second_layer_edge(cube.SIDE_IDX_LEFT,"right",1)   #right edge on left side = edge #2 +4
    if "34" == cmd:
      Cube.swap_second_layer_edge(cube.SIDE_IDX_BACK,"right",1)   #right edge on back side = edge #3 +4
    
    if "41" == cmd:
      Cube.position_yellow_edge(0)   #call on dot and line
    if "42" == cmd:
      Cube.position_yellow_edge(1)   #call on L
    if "43" == cmd:
      Cube.position_yellow_edge(2)   
    if "44" == cmd:
      Cube.position_yellow_edge(3)   

    if "51" == cmd:
      Cube.swap_yellow_edge(0)   
    if "52" == cmd:
      Cube.swap_yellow_edge(1)   
    if "53" == cmd:
      Cube.swap_yellow_edge(2)   
    if "54" == cmd:
      Cube.swap_yellow_edge(3)   

    if "61" == cmd:
      Cube.position_yellow_corner(0)   
    if "62" == cmd:
      Cube.position_yellow_corner(1)   
    if "63" == cmd:
      Cube.position_yellow_corner(2)   
    if "64" == cmd:
      Cube.position_yellow_corner(3)   

    if "71" == cmd:
      Cube.rotate_yellow_corner(0,1)     
    if "72" == cmd:
      Cube.rotate_yellow_corner(1,1)     
    if "73" == cmd:
      Cube.rotate_yellow_corner(2,1)     
    if "74" == cmd:
      Cube.rotate_yellow_corner(3,1)     

    if "u" == cmd: #top
      Cube.actions_simple(0)
    if "u'" == cmd:
      Cube.actions_simple(6)
    if "d" == cmd: #down
      Cube.actions_simple(1)
    if "d'" == cmd:
      Cube.actions_simple(7)
    if "l" == cmd: #left
      Cube.actions_simple(2)
    if "l'" == cmd:
      Cube.actions_simple(8)
    if "r" == cmd: #right
      Cube.actions_simple(3)
    if "r'" == cmd:
      Cube.actions_simple(9)
    if "f" == cmd: #front
      Cube.actions_simple(4)
    if "f'" == cmd:
      Cube.actions_simple(10)
    if "b" == cmd: #back
      Cube.actions_simple(5)
    if "b'" == cmd:
      Cube.actions_simple(11)

    console_clear()


"""
generic bruteforce search with iterative deepening (like on bruteforce-testing)
for easy algorithms with max len 6
"""
def bruteforce(Cube, itv_deep, solution):
  orig_cube = cube.tRubikCube()
  orig_edge = orig_cube.get_edge()
  orig_edge_sort = orig_cube.get_edge(sort=True)
  orig_corner = orig_cube.get_corner()

  #starting edge
  actual_edge = Cube.get_edge().copy()
  actual_corner = Cube.get_corner().copy()

  num_actions = Cube.num_actions()
  iterator_start = 0
  for search_depth in range(1, itv_deep):
    #print("Searching at %d" % search_depth)    
    seq_iterator = itertools.tIterate(search_depth, num_actions, iterator_start)
    iter_steps   = seq_iterator.get_total_num()
    iter_func    = seq_iterator.generator()

    #try all sequences and compare with original cube
    for iter in range(iter_steps):
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
        continue
      
      Test = deepcopy(Cube)  
      #Perform all rotate actions, iteration_step is a list of actions, list comprehension is faster
      [Test.actions_simple(action) for action in iteration_step]
      
      
      edge_block    = Test.get_edge()
      corner_block  = Test.get_corner()

      #white cross, solve edge per edge
      #this is very short sequence can be bruteforced, no need to use the algo rot_white_edge()
      if solution == 11:
        if edge_block[:1] == orig_edge[:1]:
          #print("Edge 0 solved")
          return deepcopy(Test) 
      if solution == 12:
        if edge_block[:2] == orig_edge[:2]:
          #print("Edge 0/1 solved")
          return deepcopy(Test) 
      if solution == 13:
        if edge_block[:3] == orig_edge[:3]:
          #print("Edge 0/1/2 solved")
          return deepcopy(Test) 
      if solution == 14:
        if edge_block[:4] == orig_edge[:4]:
          #print("Edge 0/1/2/3 solved")
          return deepcopy(Test) 
        
      
      #white edges, the sequence is too long for bruteforce so use the algo.
      #iterate over possible starting position and then use the rot_white_corner() 
      #rot_white_corner() may be called with 1, 3 or 5 repetions depending on corner orientation
      if solution == 21:
        if edge_block[:4] == actual_edge[:4]:
          #maybe it is solved here
          if corner_block[:1] ==  orig_corner[:1]:
            #print("Corner 0 solved")
            return deepcopy(Test)
          #try edge rotation algo with 1/3/5
          for i in range(1,6,2):
            edge_rotate = deepcopy(Test) 
            edge_rotate.rot_white_corner(cube.SIDE_IDX_LEFT, i)
            corner_block = edge_rotate.get_corner()
            if corner_block[:1] ==  orig_corner[:1]:
                #print("Corner 0 solved")
                return deepcopy(edge_rotate)

      if solution == 22:
        if edge_block[:4] == actual_edge[:4]:
          #maybe it is solved here
          if corner_block[:2] ==  orig_corner[:2]:
            #print("Corner 0/1 solved")
            return deepcopy(Test)
          #try edge rotation algo with 1/3/5
          for i in range(1,6,2):
            edge_rotate = deepcopy(Test) 
            edge_rotate.rot_white_corner(cube.SIDE_IDX_BACK, i)
            corner_block = edge_rotate.get_corner()
            if corner_block[:2] ==  orig_corner[:2]:
                #print("Corner 0/1 solved")
                return deepcopy(edge_rotate)
      
      if solution == 23:
        if edge_block[:4] == actual_edge[:4]:
          #maybe it is solved here
          if corner_block[:3] ==  orig_corner[:3]:
            #print("Corner 0/1/2 solved")
            return deepcopy(Test)
          #try edge rotation algo with 1/3/5
          for i in range(1,6,2):
            edge_rotate = deepcopy(Test) 
            edge_rotate.rot_white_corner(cube.SIDE_IDX_RIGHT, i)
            corner_block = edge_rotate.get_corner()
            if corner_block[:3] ==  orig_corner[:3]:
              #print("Corner 0/1/2 solved")
              return deepcopy(edge_rotate)

      if solution == 24:
        if edge_block[:4] == actual_edge[:4]:
          #maybe it is solved here
          if corner_block[:4] ==  orig_corner[:4]:
            #print("Corner 0/1/2 solved")
            return deepcopy(Test)
          #try edge rotation algo with 1/3/5 repetitions
          for i in range(1,6,2):
            edge_rotate = deepcopy(Test) 
            edge_rotate.rot_white_corner(cube.SIDE_IDX_FRONT, i)
            corner_block = edge_rotate.get_corner()
            if corner_block[:4] ==  orig_corner[:4]:
              #print("Corner 0/1/2/3 solved")
              return deepcopy(edge_rotate)

  #no solution found
  return None 



def solve_second_layer_edge(Cube, edge_num):
  orig_cube = cube.tRubikCube()
  orig_edge = orig_cube.get_edge()
  orig_edge_sort = orig_cube.get_edge(sort=True)
  
  #unsorted edge values represent the 
  target_edge = orig_cube.get_edge(edge_num, sort=False)


  #print("<<<<Solve Second layer Edge e%02d>>>>" %edge_num)
  _edge_location   = Cube.search_edge(orig_edge)
  #print(" All Edge location: %s " % _edge_location)
  edge_location = Cube.search_edge(orig_edge[edge_num])
  #print(" Edge Num e%02d is at location e%02d" %(edge_num, edge_location))
  
  #edge on upper side, major error
  if(edge_location < 4):
    return

  map_edge_num_to_side_right_algo = {
    4 : cube.SIDE_IDX_RIGHT,   #right edge on right side = edge #0 +4
    5 : cube.SIDE_IDX_FRONT,
    6 : cube.SIDE_IDX_LEFT,
    7 : cube.SIDE_IDX_BACK
    }
     
  #edge is in 2nd layer but on wrong spot, turn it on 3rd layer by applying "swap" once
  if(edge_location < 8) and (edge_location != edge_num):
    #print("  Edge Num e%02d is in 2nd layer on wrong spot turn it out..."  % (edge_num))
    Cube.swap_second_layer_edge(map_edge_num_to_side_right_algo[edge_location],"right",1)   
    edge_location = Cube.search_edge(orig_edge[edge_num]) #new edge location

    _edge_location   = Cube.search_edge(orig_edge)
    #print("All Edge location: %s " % _edge_location) 
    
  #if edge is not on the right spot
  if(edge_location != edge_num):
    #now rotate the down-side and try if we can bring the 2nd layer edge on the correct spot
    for i in range (4):
      test_cube = deepcopy(Cube)
      #rotate down-side CW, 0x, 1x, 2x 3x times (3x equals to CCW)
      if(i < 3): 
        for n in range(i): test_cube.actions_simple(1)  
      else:      
        test_cube.actions_simple(7)

      test_cube.swap_second_layer_edge(map_edge_num_to_side_right_algo[edge_num],"right",1)   #apply algo once
      edge_location = test_cube.search_edge(orig_edge[edge_num])
      _edge_location   = test_cube.search_edge(orig_edge)
     #print("  try %d: All Edge location: %s " % (i, _edge_location))

      if(edge_location == edge_num):
        Cube = deepcopy(test_cube)
        #print("  Edge Num e%02d SOLVED (is now at location e%02d)" %(edge_num, edge_location))
        _edge_location   = Cube.search_edge(orig_edge)
        #print("  Actual All Edge location: %s " % _edge_location)
        break
  
  #edge is on the right spot now
  #get the unsorted values
  actual_edge = Cube.get_edge(edge_num, sort=False)
  #print("Actual Rotation: %s, Target Rotation: %s" %(actual_edge, target_edge))
  if(actual_edge != target_edge):
    #print("Rotation is not correct, apply algo...")
    Cube.swap_second_layer_edge(map_edge_num_to_side_right_algo[edge_num], "right", 1)
    Cube.actions_simple(1)
    Cube.actions_simple(1)
    Cube.swap_second_layer_edge(map_edge_num_to_side_right_algo[edge_num], "right", 1)
  #now must be correct
  return Cube

#TODO: this stuff does not work!!!!
#search for patterns on the yellow side and perform algo according
def solve_yellow_cross(Cube):
  #get only the down-side color information
  #get color index from center block of downe-side
  color_yellow_idx = cube.COL_IDX_YELLOW

  dot_pattern = [False,False,False,False]   

  line_patterns = [
    [False,True ,False,True ],   #Horiz    line   
    [True ,False,True ,False],   #Vertical line
    ]
  
  edge_patterns = [
    [True ,False,False,True ],  #left-back        
    [True ,True ,False,False],  #back-right  
    [False,True ,True ,False],  #right-front     
    [True ,True ,False,False]   #front-left
    ]
  
  solved = [True, True, True, True]
  
  #2 possible sequences are needed 0/1 or 2/3 depending on starting state.
  #we do noit check starting state so we try both
  algo_sequences =[[0,1],[2,3]]
  for algo_seq in algo_sequences:
    yellow_edge = [0] * 4
    yellow_edge[0] = Cube.get_edge(8)[0]==color_yellow_idx
    yellow_edge[1] = Cube.get_edge(9)[0]==color_yellow_idx
    yellow_edge[2] = Cube.get_edge(10)[0]==color_yellow_idx
    yellow_edge[3] = Cube.get_edge(11)[0]==color_yellow_idx
    if(yellow_edge == solved): 
      #print("Yellow Cross already solved")
      break

    algo = 0
    cnt = 0
    for i in range(6):
      #create an array of booleans, True if color is yellow
      yellow_edge = [0] * 4
      yellow_edge[0] = Cube.get_edge(8)[0]==color_yellow_idx
      yellow_edge[1] = Cube.get_edge(9)[0]==color_yellow_idx
      yellow_edge[2] = Cube.get_edge(10)[0]==color_yellow_idx
      yellow_edge[3] = Cube.get_edge(11)[0]==color_yellow_idx
      #print("%s" % (yellow_edge))

      if(yellow_edge == solved): 
        #print("Yellow Cross Solved (%d algos)" % cnt)
        #Cube.print_2d()
        break
      #we call both algos in changing order, this will solve each cube within 5 loops
      #it is like calling the same algo with D2 between (180° turn of down-side)
        #this algo has a degree of 6
      #sometimes the first calls make a back-step from L-shape to a dot-shape
      #because we do not check the starting condition and may apply the wrong starting algo
      Cube.position_yellow_edge(algo_seq[algo])
      if(algo == 0): algo = 1
      else:          algo = 0
      cnt += 1
      #Cube.print_2d()
      #input()



    """
    #would be more effiecient to perform the correct algo depending on the pattern found
    #also use shortcut algo for L to solved direct
    #2 algos are missing
      if(yellow_edge == dot_pattern):
        print(" 'Dot' Pattern found")
        Cube.position_yellow_edge(0)
    
      if(yellow_edge == line_patterns[0]):
        print(" 'Line' Pattern H found")
        Cube.position_yellow_edge(0)    
      if(yellow_edge == line_patterns[1]):
        print(" 'Line' Pattern V found")
    
      if(yellow_edge == edge_patterns[0]):
        print(" 'L' Pattern found, left/back")
        Cube.position_yellow_edge(0)
      if(yellow_edge == edge_patterns[1]):
        print(" 'L' Pattern found, back/right")
      if(yellow_edge == edge_patterns[2]):
        print(" 'L' Pattern found, right/front")
        Cube.position_yellow_edge(1)    #--> H-Line

      if(yellow_edge == edge_patterns[3]):
        print(" 'L' Pattern found, front/left")
    """
  return Cube

#solve the yellow edges, orient their adjacent sides.
def solve_yellow_edge(Cube):
      #0:  e10-e11 
      #1:  e8-e9 
      #2:  e8-e11 
      #3:  e9-e10 
  orig_cube = cube.tRubikCube()
  orig_edge = orig_cube.get_edge()
  orig_edge_sort = orig_cube.get_edge(sort=True)
  #_edge_location   = Cube.search_edge(orig_edge)
  #print(" All Edge location: %s " % _edge_location)
    
  #rotate down until e8 solved, must be possible if e8 is on yellow side
  edge_num = 8
  edge_location = Cube.search_edge(orig_edge[edge_num])
  if(edge_location < 8): 
    print("e%02d unsolveable" %edge_num)
    return

  #print("<<<<Solve 3rd layer Edge e%02d>>>>" %edge_num)
  for i in range (4):
    test_cube = deepcopy(Cube)
    #rotate down-side CW, 0x, 1x, 2x 3x times (3x equals to CCW)
    if(i < 3): 
      for n in range(i): test_cube.actions_simple(1)  
    else:      
      test_cube.actions_simple(7)
    #find edge 8
    edge_location = test_cube.search_edge(orig_edge[edge_num])
    #print(" Edge Num e%02d is at location e%02d" %(edge_num, edge_location))
    if(edge_location == edge_num): 
      Cube = deepcopy(test_cube)
      break
  #_edge_location   = Cube.search_edge(orig_edge)
  #print(" All Edge location: %s " % _edge_location)
  
  #solve edge 9
  edge_num = 9
  edge_location = Cube.search_edge(orig_edge[edge_num])
  if(edge_location < 8): 
    print("e%02d unsolveable" %edge_num)
    return
  #print("<<<<Solve 3rd layer Edge e%02d>>>>" %edge_num)
  #can be 9, 10, or 11
  #9 is at 10 is a simple swap
  if (edge_location == 10):
    Cube.swap_yellow_edge(3)
  #9 is at 11 means they are opposite - use combination of 2x algo
  if (edge_location == 11):
    Cube.actions_simple(1)  
    Cube.swap_yellow_edge(0)
    Cube.actions_simple(1)  
    Cube.actions_simple(1)  
    Cube.swap_yellow_edge(0)
    Cube.actions_simple(1)  
    Cube.actions_simple(1)  
  #_edge_location   = Cube.search_edge(orig_edge)
  #print(" All Edge location: %s " % _edge_location)

  #solve edge 10
  edge_num = 10
  edge_location = Cube.search_edge(orig_edge[edge_num])
  if(edge_location < 8): 
    print("e%02d unsolveable" %edge_num)
    return
  #print("<<<<Solve 3rd layer Edge e%02d>>>>" %edge_num)
  #can be 10 or 11, simple swap, will only be 11 if e9 was at 10
  if (edge_location == 11):
    Cube.swap_yellow_edge(0)
  #_edge_location   = Cube.search_edge(orig_edge)
  #print(" All Edge location: %s " % _edge_location)
  return Cube

#
def position_yellow_corner(Cube):
  orig_cube = cube.tRubikCube()
  orig_corner = orig_cube.get_corner()
  orig_corner_sort = orig_cube.get_corner(sort=True)
  
  #_corner_location   = Cube.search_corner(orig_corner)
  #print(" All Corner location: %s " % _corner_location)
  
  #search for a corner on the correct spot, sometimes no corner is on correct spot
  any_corner_correct = False
  for i in range(4):
    corner_num = 4+i
    corner_location = Cube.search_corner(orig_corner[corner_num])
    if(corner_location == corner_num):
      any_corner_correct = True
      break
  

  #we didn't find a correct corner, we lock corner 7
  if(any_corner_correct == False):
    corner_location = 7
    #print(" No corner at correct spot! lock Corner c%02d and loop until one is correct" % corner_location)

    corner_lock_algo_num = corner_location-4
    #repeat algo until we have at least one correct corner
    while True:
      Cube.position_yellow_corner(corner_lock_algo_num)
      #we test c4,c5,c6 - c7 is locked and will not change
      any_corner_correct = False
      for i in range(3):
        corner_num = 4+i
        corner_location = Cube.search_corner(orig_corner[corner_num])
        if(corner_location == corner_num):
          any_corner_correct = True
          break
      if any_corner_correct == True:
        #print("Found a corner on correct spot")
        break
    #_corner_location   = Cube.search_corner(orig_corner)
    #print(" All Corner location: %s " % _corner_location)
  
  #now we have at least one corner in the correct spot.
  #now we lock this corner and repeat the algo until all corners correct
  if any_corner_correct == True:
    target_corner = orig_corner_sort[4:8]
    actual_corner = Cube.get_corner(sort=True)[4:8]
    #maybe all is correct, we can return immediately
    if(actual_corner==target_corner):
      return Cube

    corner_lock_algo_num = corner_location-4
    #print(" Corner c%02d at correct spot! lock it" % corner_location)
    #_corner_location   = Cube.search_corner(orig_corner)
    #print(" All Corner location: %s " % _corner_location)

    while True:
      Cube.position_yellow_corner(corner_lock_algo_num)
      actual_corner = Cube.get_corner(sort=True)[4:8]
      #_corner_location   = Cube.search_corner(orig_corner)
      #print("  All Corner location: %s " % _corner_location)
      if(actual_corner==target_corner):
        break

  #_corner_location   = Cube.search_corner(orig_corner)
  #print(" All Corner location: %s " % _corner_location)

  #mapping algo num to corner lock algo:
      #0: #c4 unchanged
      #1: #c5 unchanged
      #2: #c6 unchanged
      #3: #c7 unchanged
  return Cube

def orient_last_layer_corner(Cube):
  orig_cube = cube.tRubikCube()
  orig_corner = orig_cube.get_corner()

  color_yellow_idx = cube.COL_IDX_YELLOW
  solved = [True, True, True, True]
  
  #probe yellow corners
  yellow_corner = [None] * 4
  for i in range(4):
    yellow_corner[i] = Cube.get_corner(i+4)[0]==color_yellow_idx

  #is the puzzle finished?
  if(solved == yellow_corner): 
    #print("Finished - Nothing more to do")
    return Cube
  
  #get starting corner (first misplaced)
  corner_num = 4
  for i in range(4):
    if(yellow_corner[i] == False): break
    corner_num += 1
  wrong_corners = yellow_corner.count(False)

  #now corner num points to the first misplaced corner
  algo_num = corner_num - 4 
  #print(" %d Wrong corners,  Start at c%02d, use algo num: %d " % (wrong_corners, corner_num, algo_num))

  while True:
    #print(" Orient Last Layer corner...")
    #apply algo up to 2 times - with 2 repetitions
    while True:
      actual_corner = Cube.get_corner(corner_num)[0]==color_yellow_idx
      if(actual_corner == False):
        Cube.rotate_yellow_corner(algo_num, 2)
        #Cube.print_2d()
        #input()
      else: break

    #probe yellow corners
    yellow_corner = [None] * 4
    for i in range(4):
      yellow_corner[i] = Cube.get_corner(i+4)[0]==color_yellow_idx
    #is yellow side finished?
    if(solved == yellow_corner): 
      break;
    wrong_corners = yellow_corner.count(False)
    #print(" %d Wrong corners" % wrong_corners)

    #there are still wrong corners - rotate cube until next wrong corner is in spot
    while True:
      Cube.actions_simple(1) 
      actual_corner = Cube.get_corner(corner_num)[0]==color_yellow_idx
      #Cube.print_2d()
      #now the next wrong corner is in spot
      if actual_corner == False: break

  
  #here it might be necessary to rotate the down-side
  for i in range (4):
    if(Cube.equals(orig_cube) == True):
      #print("Finished - Nothing more to do")
      return Cube
    else:
      Cube.actions_simple(1) 



  return Cube


if __name__=="__main__":
  main()

