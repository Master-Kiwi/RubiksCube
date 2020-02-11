#CHANGELOG: RubiksCube.py
#AUTHOR: SL

#TODO: rework scoring functions:
#  implement methode for reading corner color []*3 and edge color []*2 from cube arrays
#    is needed in selftest and in scoring
#  implement new scoring based on:
#    corner position and rotation
#    edge position and rotation  
#  test this new scoring
#  unify algo methods, maybe create a dict for all algos and use a single method to call
#     move algos to new file "Rubiks_Cube_algos.py"?
#  add method for printing sequences in letters instead of numbers (R / R')
#  print letter sequence also to file additional to numbers
#  unify notation, change TOP/BOT to UP/DOWN. this is common notation on internet and also less sensitive to errors
#     actions-shortcuts are in this case UNIQUE single letters with this notation
#     BOT/BACK > new notation: D(down) / B(back)
#  add metrics (actual is quarter-turn), add half-turn (maybe decide with constructor what metric to use)

#optional:
#  add some functions to create .png or .jpg files of cube states

#11.02.2020 
#  solved: "implement print function to visualize edge/corner, maybe print_2d_ext() that outputs the corner / edges right beside the cube" from TODO
#  solved: "implement standard cube algos from traditional 6-step solution" from TODO
#   -rot_white_edge()
#   -rot_white_corner()
#   ....
#  updated: self_test() to work with new corner/edge/center methods
#  updated: actions_simple() - removed many dead code
#  added: new methods to tRubikCube Class:
#   -get_corner(location=None, sort=False)          return the color value of specific or all corner-blocks [i,j,k]
#   -get_edge(location=None, sort=False)            return the color value of specific or all edge-blocks [i,j]
#   -get_center(location=None):                     return the color value of specific or all center-blocks
#   -search_corner(edge_to_search = num or list):   return the position(s) of corner(s) - search by value, will ignore rotation
#   -search_edge(edge_to_search = num or list):     return the position(s) of edge(s)   - search by value, will ignore rotation

 

#12.01.2020 
#  improved print_2d(), now also fills the boundary box with black bg
#  class tRubikCube simplified, moved dicts / lists outside. class now carries only the .col data and action_list


#04.01.2020 
#improved performance on rotation
#added methods to tRubikCube Class:
# -rotate_simple:   simple version of rotate() with direct modification of self.col array instead of numpy operations
# -actions_simple:  simple version of actions()
# -score:           simple scoring to get a numeric value hiw "finished" the cube is compared to the original-state
# -compare:         similar to score but compares self.col to another cube data. if compare() is called with original-state-cube its result is identical with score()
#   with the last results from test of score and compare: re-think about the definition of score() and compare()
#     treat edges and corners different in computation of score. edge_pos_score and edge_rot_score
#     edge can have 2 rotations and 12 positions
#     corner can have 6? (or 3?) rotations and 8 positions
#   iterations/sec is still an issue, with multiprocessing we have up to 30k/sec. at depth=8 we have 429M iterations to compute what needs ~4hours.
#   we will not find a solution here for a prerotated cube with wrong corners at this depth of 8
#   implement further improvement of actions() function: (without tree search)
#     with each new action() save state of self.col to a new list of already_seen states. 
#     check if the new self.col has been seen before, if yes we can skip all further actions as this will no be the path to the fastest solution
#     will consume less memory, as we deal at the moment only with depths < 12, and therefore action_lists < 12


#added various testing functions:
# -cube_actions_check()
# -cube_benchmark()
# -cube_score_check()
# example in main()


#27.12.2019
#initial version 
#  class tRubikCube
#   self.col[6][3][3] contains the color data (index) - that represents the state
#basic methods for 3x3 cube:
# -print2d:   colored output to console
# -rotate:    basic rotate commands for each side
# -turn:      basic command to change the "facing" side, used for validation of rotation
# -selftest:  basic self-check
# -actions:   maps action number to rotate actions (0 -> TOP /CW)
# -equals:    checks self.col of 2 instances
# -save_to_file:    save to JSON File
# -load from file   load from JSON File
# -various internal helper methods


import os 
import numpy as np
import json
import datetime
from copy import copy, deepcopy
import random
from helpers import console_clear

SIDE_IDX_TOP    = 0
SIDE_IDX_BOT    = 1
SIDE_IDX_FRONT  = 2
SIDE_IDX_BACK   = 3
SIDE_IDX_LEFT   = 4
SIDE_IDX_RIGHT  = 5

#do not change those numbers - referenced in selftest
COL_IDX_WHITE   = 0
COL_IDX_YELLOW  = 1 
COL_IDX_ORANGE  = 2
COL_IDX_RED     = 3
COL_IDX_GREEN   = 4
COL_IDX_BLUE    = 5
COL_IDX_BLACK   = 6
COL_IDX_END     = 7

ROT_DIR_CW      = True
ROT_DIR_CCW     = False

TURN_DIR_LEFT_TO_TOP   = 1    #axis = Y
TURN_DIR_RIGHT_TO_TOP  = 2
TURN_DIR_FRONT_TO_TOP  = 3    #axis = X
TURN_DIR_BACK_TO_TOP   = 4
TURN_DIR_LEFT_TO_FRONT = 5    #axis = Z
TURN_DIR_RIGHT_TO_FRONT= 6

#color format strings, check them using console_print_color_table() from helpers.py
#col_str_black   = '\x1b[1;37;40m'     #for test use 1;37;46m
col_str_black   = '\x1b[1;37;46m'     #for test use 1;37;46m
col_str_red     = '\x1b[1;37;41m'
col_str_green   = '\x1b[1;37;42m'
col_str_yellow  = '\x1b[1;37;43m'
col_str_blue    = '\x1b[1;37;44m'
col_str_orange  = '\x1b[1;37;45m'
col_str_white   = '\x1b[1;37;47m'
col_str_end     = '\x1b[0m'

#just for print2d, us a dict
col_fmt_str = { 
  COL_IDX_WHITE   : col_str_white,  
  COL_IDX_YELLOW  : col_str_yellow,  
  COL_IDX_ORANGE  : col_str_orange,  
  COL_IDX_RED     : col_str_red,  
  COL_IDX_GREEN   : col_str_green,  
  COL_IDX_BLUE    : col_str_blue,    
  COL_IDX_BLACK   : col_str_black,  
  COL_IDX_END     : col_str_end,  
  }


#self.action_dict lists all possible actions, key=action_idx
action_dict = { 
  0: "ROTATE TOP / CW",
  6: "ROTATE TOP / CCW",
  1: "ROTATE BOT / CW",
  7: "ROTATE BOT / CCW",
  2: "ROTATE LEFT / CW",
  8: "ROTATE LEFT / CCW",  
  3: "ROTATE RIGHT / CW",
  9: "ROTATE RIGHT / CCW",
  4: "ROTATE FRONT / CW",
  10:"ROTATE FRONT / CCW",
  5: "ROTATE BACK / CW",
  11:"ROTATE BACK / CCW"
}
#key=color index
color_dict = {
  COL_IDX_WHITE   : "white",
  COL_IDX_YELLOW  : "yellow", 
  COL_IDX_ORANGE  : "orange",
  COL_IDX_RED     : "red",
  COL_IDX_GREEN   : "green",
  COL_IDX_BLUE    : "blue"
}
#key=side index
side_dict = {
  SIDE_IDX_TOP    : "top",
  SIDE_IDX_BOT    : "bot",
  SIDE_IDX_FRONT  : "front",
  SIDE_IDX_BACK   : "back",
  SIDE_IDX_LEFT   : "left",
  SIDE_IDX_RIGHT  : "right"
}


class tRubikCube:
  N_DIM: int      #dimension of cube (=side len)
    
  def __init__(self):
    #actual only 3 is supported
    self.N_DIM = 3 
    
    #start with empty action list
    #stores all actions that are performed on this object. (by index)
    self.actions_list = []
   
    #set initial state of cube = solved cube
    #opposite sides
    self.col = np.zeros([6,self.N_DIM,self.N_DIM], dtype=int)
    self.col[SIDE_IDX_TOP]    = COL_IDX_WHITE
    self.col[SIDE_IDX_BOT]    = COL_IDX_YELLOW
    self.col[SIDE_IDX_FRONT]  = COL_IDX_ORANGE
    self.col[SIDE_IDX_BACK]   = COL_IDX_RED  
    self.col[SIDE_IDX_LEFT]   = COL_IDX_BLUE
    self.col[SIDE_IDX_RIGHT]  = COL_IDX_GREEN
    
    #self.col[SIDE_IDX_LEFT][0][0]  = COL_IDX_RED
    #self.col[SIDE_IDX_RIGHT][0][0]  = COL_IDX_RED
    #self.col[SIDE_IDX_TOP][0][0]  = COL_IDX_GREEN
    #self.col[SIDE_IDX_BOT][0][0]  = COL_IDX_GREEN
    #self.col[SIDE_IDX_BACK][0][0]  = COL_IDX_WHITE
    #self.col[SIDE_IDX_FRONT][0][0]  = COL_IDX_WHITE

  
  #for colored console output, prints <num_blocks> spaces with col_idx as fg/bg/style
  #looks like large colored pixels
  #changed to read from dict
  def _print_blocks(self, col_idx, num_blocks):
    print(col_fmt_str[col_idx] + (' '*num_blocks) + col_fmt_str[COL_IDX_END], sep='', end='')
  #for colored console output, prints text spaces with col_idx as fg/bg
  #pay attention that fg color is set correct in the col_idx strings
  def _print_col_text(self, col_idx, col_text):
    print(col_fmt_str[col_idx] + col_text + col_fmt_str[COL_IDX_END], sep='', end='')
  
  #print additional information edges and colors, called from print_2d() only, but may be called individually
  def _print_ext_info(self, edge_block, corner_block, num):
    ilen = 3  #a cube side col block has a len of 3
    self._print_col_text(COL_IDX_BLACK, "e%02d  " % num)
    self._print_blocks(edge_block[0],ilen)
    self._print_blocks(edge_block[1],ilen)
    self._print_blocks(COL_IDX_BLACK,ilen*4)
    
    if not corner_block:
      self._print_col_text(COL_IDX_BLACK, " " * (5+ilen*4))
    else:
      #corner
      self._print_col_text(COL_IDX_BLACK, "c%02d  " % num)
      self._print_blocks(corner_block[0], ilen)
      self._print_blocks(corner_block[1], ilen)
      self._print_blocks(corner_block[2], ilen)
      self._print_blocks(COL_IDX_BLACK,ilen)
      
  #print 2D view of the cube on console    
  #it was changed due to weird color output to fill the whole cube-boundary box with black boxes
  #additional information added, edges and colors, mapping according to cube_col_array_index_map.xlsx
  def print_2d(self):

    edge_block = self.get_edge()
    #center_block = self.get_center()
    corner_block = self.get_corner()

    ilen = 3  #a cube side col block has a len of 3
    print('', end='\n')
    self._print_blocks(COL_IDX_BLACK, ilen*19)    #fill first line
    self._print_col_text(COL_IDX_BLACK, "edges                  corners          ")

    for i in range (self.N_DIM):
      print('', end='\n')
      self._print_blocks(COL_IDX_BLACK, ilen*5)
      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_BACK][i][j], ilen)
      self._print_blocks(COL_IDX_BLACK, ilen*11)
      self._print_ext_info(edge_block[i], corner_block[i], i)

    print('', end='\n')
    self._print_blocks(COL_IDX_BLACK, ilen*19)    #fill seperator back/top
    self._print_ext_info(edge_block[3], corner_block[3], 3)

    for i in range (self.N_DIM):
      print('', end='\n')
      self._print_blocks(COL_IDX_BLACK, ilen)

      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_LEFT][i][j], ilen)
      
      self._print_blocks(COL_IDX_BLACK, ilen)
      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_TOP][i][j], ilen)
      
      self._print_blocks(COL_IDX_BLACK, ilen)      
      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_RIGHT][i][j], ilen)
      
      self._print_blocks(COL_IDX_BLACK, ilen)
      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_BOT][i][j], ilen)

      self._print_blocks(COL_IDX_BLACK, ilen*3)   #fill last row
      self._print_ext_info(edge_block[i+4], corner_block[i+4], i+4)

    print('', end='\n')
    self._print_blocks(COL_IDX_BLACK, ilen*19)  #fill seperator top/front
    self._print_ext_info(edge_block[7], corner_block[7], 7) #last corner idx

    for i in range (self.N_DIM):
      print('', end='\n')
      self._print_blocks(COL_IDX_BLACK, ilen*5)
      for j in range (self.N_DIM):
        self._print_blocks(self.col[SIDE_IDX_FRONT][i][j], ilen)        
      self._print_blocks(COL_IDX_BLACK, ilen*11)
      self._print_ext_info(edge_block[8+i], None, 8+i)

    print('', end='\n')
    self._print_blocks(COL_IDX_BLACK, ilen*19)  #fill last line
    self._print_ext_info(edge_block[11], None, 11)
    print('', end='\n')

  
  def _rotate_side(self, side_idx, rotate_dir=ROT_DIR_CW):   
    #np.rot90(<array>, 1) rotates array by 90 degrees in CCW
    #to have -90degrees we just rotate 3-times, what means a CW rotation
    #rotate_dir = True is CW rotation
    if rotate_dir:
      self.col[side_idx] = np.rot90(self.col[side_idx] , 3)
    else:
      self.col[side_idx] = np.rot90(self.col[side_idx] , 1)

  #rotate the adjacent sides, extract rows, rotate and write back
  def _rotate_adjacent_sides(self, upper, right, lower, left, rotate_dir):     
    #always same rotation, as adjacent sides have to be presented correct from calling function
    right = np.rot90(right, 1)
    lower = np.rot90(lower, 2)
    left  = np.rot90(left, 3)

    #get the 4 rows to rotate 
    row =  [[0]*self.N_DIM for _ in range(4)]     #create empty list
    row[0] = upper[2].copy()
    row[1] = right[2].copy()
    row[2] = lower[2].copy()
    row[3] = left[2].copy()
      
    mem_row=row.copy() #remember state before rotation
      
    if rotate_dir == ROT_DIR_CW:
      #print("  Direction: Clockwise")
      row[0] = mem_row[3]
      row[1] = mem_row[0]
      row[2] = mem_row[1]
      row[3] = mem_row[2]

    if rotate_dir == ROT_DIR_CCW:
      #print("  Direction: Counter-Clockwise")
      row[0] = mem_row[1]
      row[1] = mem_row[2]
      row[2] = mem_row[3]
      row[3] = mem_row[0]

    upper[2] = row[0].copy()
    right[2] = row[1].copy()
    lower[2] = row[2].copy()
    left[2]  = row[3].copy()

    #upper is correct
    right = np.rot90(right, 3)
    lower = np.rot90(lower, 2)
    left  = np.rot90(left, 1)
 
  
  def rotate_simple(self, side_idx, rotate_dir=ROT_DIR_CW):
    #rotate the main side, just a array rotatation
    self._rotate_side(side_idx, rotate_dir)
    if side_idx == SIDE_IDX_TOP: 
      mem_side_back = self.col[SIDE_IDX_BACK][2].copy()
      #if rotate_dir == ROT_DIR_CW:    #0 slower!
      #  i=0
      #  j=2
      #  while i < 3:
      #    self.col[SIDE_IDX_BACK][2][j]   = self.col[SIDE_IDX_LEFT][i][2] 
      #    self.col[SIDE_IDX_LEFT][i][2]   = self.col[SIDE_IDX_FRONT][0][i] 
      #    self.col[SIDE_IDX_FRONT][0][i]  = self.col[SIDE_IDX_RIGHT][j][0]
      #    self.col[SIDE_IDX_RIGHT][j][0]  = mem_side_back[j]
      #    i += 1
      #    j -= 1
      if rotate_dir:    #0    #True = CW
        self.col[SIDE_IDX_BACK][2][0]   = self.col[SIDE_IDX_LEFT][2][2] 
        self.col[SIDE_IDX_BACK][2][1]   = self.col[SIDE_IDX_LEFT][1][2]
        self.col[SIDE_IDX_BACK][2][2]   = self.col[SIDE_IDX_LEFT][0][2]
        self.col[SIDE_IDX_LEFT][0][2]   = self.col[SIDE_IDX_FRONT][0][0] 
        self.col[SIDE_IDX_LEFT][1][2]   = self.col[SIDE_IDX_FRONT][0][1] 
        self.col[SIDE_IDX_LEFT][2][2]   = self.col[SIDE_IDX_FRONT][0][2] 
        self.col[SIDE_IDX_FRONT][0][0]  = self.col[SIDE_IDX_RIGHT][2][0]
        self.col[SIDE_IDX_FRONT][0][1]  = self.col[SIDE_IDX_RIGHT][1][0]
        self.col[SIDE_IDX_FRONT][0][2]  = self.col[SIDE_IDX_RIGHT][0][0]
        self.col[SIDE_IDX_RIGHT][0][0]  = mem_side_back[0] 
        self.col[SIDE_IDX_RIGHT][1][0]  = mem_side_back[1]
        self.col[SIDE_IDX_RIGHT][2][0]  = mem_side_back[2]
      else:   #6
        self.col[SIDE_IDX_BACK][2][0]   = self.col[SIDE_IDX_RIGHT][0][0] 
        self.col[SIDE_IDX_BACK][2][1]   = self.col[SIDE_IDX_RIGHT][1][0] 
        self.col[SIDE_IDX_BACK][2][2]   = self.col[SIDE_IDX_RIGHT][2][0] 
        self.col[SIDE_IDX_RIGHT][0][0]  = self.col[SIDE_IDX_FRONT][0][2]
        self.col[SIDE_IDX_RIGHT][1][0]  = self.col[SIDE_IDX_FRONT][0][1]
        self.col[SIDE_IDX_RIGHT][2][0]  = self.col[SIDE_IDX_FRONT][0][0]
        self.col[SIDE_IDX_FRONT][0][0]  = self.col[SIDE_IDX_LEFT][0][2]
        self.col[SIDE_IDX_FRONT][0][1]  = self.col[SIDE_IDX_LEFT][1][2]
        self.col[SIDE_IDX_FRONT][0][2]  = self.col[SIDE_IDX_LEFT][2][2]
        self.col[SIDE_IDX_LEFT][0][2]   = mem_side_back[2] 
        self.col[SIDE_IDX_LEFT][1][2]   = mem_side_back[1] 
        self.col[SIDE_IDX_LEFT][2][2]   = mem_side_back[0] 
    elif side_idx == SIDE_IDX_BOT: 
      mem_side_back = self.col[SIDE_IDX_BACK][0].copy()
      if rotate_dir:    #1
        self.col[SIDE_IDX_BACK][0][2]   = self.col[SIDE_IDX_RIGHT][2][2] 
        self.col[SIDE_IDX_BACK][0][1]   = self.col[SIDE_IDX_RIGHT][1][2] 
        self.col[SIDE_IDX_BACK][0][0]   = self.col[SIDE_IDX_RIGHT][0][2] 
        self.col[SIDE_IDX_RIGHT][2][2]  = self.col[SIDE_IDX_FRONT][2][0]
        self.col[SIDE_IDX_RIGHT][1][2]  = self.col[SIDE_IDX_FRONT][2][1]
        self.col[SIDE_IDX_RIGHT][0][2]  = self.col[SIDE_IDX_FRONT][2][2]
        self.col[SIDE_IDX_FRONT][2][0]  = self.col[SIDE_IDX_LEFT][0][0]
        self.col[SIDE_IDX_FRONT][2][1]  = self.col[SIDE_IDX_LEFT][1][0]
        self.col[SIDE_IDX_FRONT][2][2]  = self.col[SIDE_IDX_LEFT][2][0]
        self.col[SIDE_IDX_LEFT][0][0]   = mem_side_back[2] 
        self.col[SIDE_IDX_LEFT][1][0]   = mem_side_back[1] 
        self.col[SIDE_IDX_LEFT][2][0]   = mem_side_back[0] 
      else:   #7
        self.col[SIDE_IDX_BACK][0][2]   = self.col[SIDE_IDX_LEFT][0][0] 
        self.col[SIDE_IDX_BACK][0][1]   = self.col[SIDE_IDX_LEFT][1][0]
        self.col[SIDE_IDX_BACK][0][0]   = self.col[SIDE_IDX_LEFT][2][0]
        self.col[SIDE_IDX_LEFT][0][0]   = self.col[SIDE_IDX_FRONT][2][0] 
        self.col[SIDE_IDX_LEFT][1][0]   = self.col[SIDE_IDX_FRONT][2][1] 
        self.col[SIDE_IDX_LEFT][2][0]   = self.col[SIDE_IDX_FRONT][2][2] 
        self.col[SIDE_IDX_FRONT][2][0]  = self.col[SIDE_IDX_RIGHT][2][2]
        self.col[SIDE_IDX_FRONT][2][1]  = self.col[SIDE_IDX_RIGHT][1][2]
        self.col[SIDE_IDX_FRONT][2][2]  = self.col[SIDE_IDX_RIGHT][0][2]
        self.col[SIDE_IDX_RIGHT][0][2]  = mem_side_back[0] 
        self.col[SIDE_IDX_RIGHT][1][2]  = mem_side_back[1]
        self.col[SIDE_IDX_RIGHT][2][2]  = mem_side_back[2]
    elif side_idx == SIDE_IDX_LEFT: 
      mem_side_back = [0] * 3
      mem_side_back[0] = self.col[SIDE_IDX_BACK][0][0]
      mem_side_back[1] = self.col[SIDE_IDX_BACK][1][0]
      mem_side_back[2] = self.col[SIDE_IDX_BACK][2][0]
      if rotate_dir:    #2
        self.col[SIDE_IDX_BACK][0][0]   = self.col[SIDE_IDX_BOT][2][2] 
        self.col[SIDE_IDX_BACK][1][0]   = self.col[SIDE_IDX_BOT][1][2]
        self.col[SIDE_IDX_BACK][2][0]   = self.col[SIDE_IDX_BOT][0][2]
        self.col[SIDE_IDX_BOT][0][2]    = self.col[SIDE_IDX_FRONT][2][0] 
        self.col[SIDE_IDX_BOT][1][2]    = self.col[SIDE_IDX_FRONT][1][0] 
        self.col[SIDE_IDX_BOT][2][2]    = self.col[SIDE_IDX_FRONT][0][0] 
        self.col[SIDE_IDX_FRONT][0][0]  = self.col[SIDE_IDX_TOP][0][0]
        self.col[SIDE_IDX_FRONT][1][0]  = self.col[SIDE_IDX_TOP][1][0]
        self.col[SIDE_IDX_FRONT][2][0]  = self.col[SIDE_IDX_TOP][2][0]
        self.col[SIDE_IDX_TOP][0][0]    = mem_side_back[0] 
        self.col[SIDE_IDX_TOP][1][0]    = mem_side_back[1]
        self.col[SIDE_IDX_TOP][2][0]    = mem_side_back[2]
      else:   #8
        self.col[SIDE_IDX_BACK][0][0]   = self.col[SIDE_IDX_TOP][0][0] 
        self.col[SIDE_IDX_BACK][1][0]   = self.col[SIDE_IDX_TOP][1][0]
        self.col[SIDE_IDX_BACK][2][0]   = self.col[SIDE_IDX_TOP][2][0]
        self.col[SIDE_IDX_TOP][0][0]    = self.col[SIDE_IDX_FRONT][0][0] 
        self.col[SIDE_IDX_TOP][1][0]    = self.col[SIDE_IDX_FRONT][1][0] 
        self.col[SIDE_IDX_TOP][2][0]    = self.col[SIDE_IDX_FRONT][2][0] 
        self.col[SIDE_IDX_FRONT][0][0]  = self.col[SIDE_IDX_BOT][2][2]
        self.col[SIDE_IDX_FRONT][1][0]  = self.col[SIDE_IDX_BOT][1][2]
        self.col[SIDE_IDX_FRONT][2][0]  = self.col[SIDE_IDX_BOT][0][2]
        self.col[SIDE_IDX_BOT][2][2]    = mem_side_back[0] 
        self.col[SIDE_IDX_BOT][1][2]    = mem_side_back[1]
        self.col[SIDE_IDX_BOT][0][2]    = mem_side_back[2]
    elif side_idx == SIDE_IDX_RIGHT:
      mem_side_back = [0] * 3
      mem_side_back[0] = self.col[SIDE_IDX_BACK][2][2]
      mem_side_back[1] = self.col[SIDE_IDX_BACK][1][2]
      mem_side_back[2] = self.col[SIDE_IDX_BACK][0][2]
      if rotate_dir:    #3
        self.col[SIDE_IDX_BACK][2][2]   = self.col[SIDE_IDX_TOP][2][2] 
        self.col[SIDE_IDX_BACK][1][2]   = self.col[SIDE_IDX_TOP][1][2]
        self.col[SIDE_IDX_BACK][0][2]   = self.col[SIDE_IDX_TOP][0][2]
        self.col[SIDE_IDX_TOP][2][2]    = self.col[SIDE_IDX_FRONT][2][2] 
        self.col[SIDE_IDX_TOP][1][2]    = self.col[SIDE_IDX_FRONT][1][2] 
        self.col[SIDE_IDX_TOP][0][2]    = self.col[SIDE_IDX_FRONT][0][2] 
        self.col[SIDE_IDX_FRONT][0][2]  = self.col[SIDE_IDX_BOT][2][0]
        self.col[SIDE_IDX_FRONT][1][2]  = self.col[SIDE_IDX_BOT][1][0]
        self.col[SIDE_IDX_FRONT][2][2]  = self.col[SIDE_IDX_BOT][0][0]
        self.col[SIDE_IDX_BOT][0][0]    = mem_side_back[0] 
        self.col[SIDE_IDX_BOT][1][0]    = mem_side_back[1]
        self.col[SIDE_IDX_BOT][2][0]    = mem_side_back[2]
      else:    #9
        self.col[SIDE_IDX_BACK][2][2]   = self.col[SIDE_IDX_BOT][0][0] 
        self.col[SIDE_IDX_BACK][1][2]   = self.col[SIDE_IDX_BOT][1][0]
        self.col[SIDE_IDX_BACK][0][2]   = self.col[SIDE_IDX_BOT][2][0]
        self.col[SIDE_IDX_BOT][0][0]    = self.col[SIDE_IDX_FRONT][2][2] 
        self.col[SIDE_IDX_BOT][1][0]    = self.col[SIDE_IDX_FRONT][1][2] 
        self.col[SIDE_IDX_BOT][2][0]    = self.col[SIDE_IDX_FRONT][0][2] 
        self.col[SIDE_IDX_FRONT][0][2]  = self.col[SIDE_IDX_TOP][0][2]
        self.col[SIDE_IDX_FRONT][1][2]  = self.col[SIDE_IDX_TOP][1][2]
        self.col[SIDE_IDX_FRONT][2][2]  = self.col[SIDE_IDX_TOP][2][2]
        self.col[SIDE_IDX_TOP][2][2]    = mem_side_back[0] 
        self.col[SIDE_IDX_TOP][1][2]    = mem_side_back[1]
        self.col[SIDE_IDX_TOP][0][2]    = mem_side_back[2]
    elif side_idx == SIDE_IDX_FRONT:
      mem_side_top = self.col[SIDE_IDX_TOP][2].copy()
      if rotate_dir:    #4
        #self.col[SIDE_IDX_TOP][2][0]    = self.col[SIDE_IDX_LEFT][2][0]
        #self.col[SIDE_IDX_TOP][2][1]    = self.col[SIDE_IDX_LEFT][2][1]
        #self.col[SIDE_IDX_TOP][2][2]    = self.col[SIDE_IDX_LEFT][2][2]
        self.col[SIDE_IDX_TOP][2]       = self.col[SIDE_IDX_LEFT][2]
        #self.col[SIDE_IDX_LEFT][2][0]   = self.col[SIDE_IDX_BOT][2][0]
        #self.col[SIDE_IDX_LEFT][2][1]   = self.col[SIDE_IDX_BOT][2][1]
        #self.col[SIDE_IDX_LEFT][2][2]   = self.col[SIDE_IDX_BOT][2][2]
        self.col[SIDE_IDX_LEFT][2]      = self.col[SIDE_IDX_BOT][2]
        #self.col[SIDE_IDX_BOT][2][0]    = self.col[SIDE_IDX_RIGHT][2][0]
        #self.col[SIDE_IDX_BOT][2][1]    = self.col[SIDE_IDX_RIGHT][2][1]
        #self.col[SIDE_IDX_BOT][2][2]    = self.col[SIDE_IDX_RIGHT][2][2]
        self.col[SIDE_IDX_BOT][2]       = self.col[SIDE_IDX_RIGHT][2]
        #self.col[SIDE_IDX_RIGHT][2][0]  = mem_side_top[0]
        #self.col[SIDE_IDX_RIGHT][2][1]  = mem_side_top[1]
        #self.col[SIDE_IDX_RIGHT][2][2]  = mem_side_top[2]
        self.col[SIDE_IDX_RIGHT][2]     = mem_side_top
      else:    #10
        #self.col[SIDE_IDX_TOP][2][0]    = self.col[SIDE_IDX_RIGHT][2][0]
        #self.col[SIDE_IDX_TOP][2][1]    = self.col[SIDE_IDX_RIGHT][2][1]
        #self.col[SIDE_IDX_TOP][2][2]    = self.col[SIDE_IDX_RIGHT][2][2]
        self.col[SIDE_IDX_TOP][2]       = self.col[SIDE_IDX_RIGHT][2]
        #self.col[SIDE_IDX_RIGHT][2][0]  = self.col[SIDE_IDX_BOT][2][0]
        #self.col[SIDE_IDX_RIGHT][2][1]  = self.col[SIDE_IDX_BOT][2][1]
        #self.col[SIDE_IDX_RIGHT][2][2]  = self.col[SIDE_IDX_BOT][2][2]
        self.col[SIDE_IDX_RIGHT][2]     = self.col[SIDE_IDX_BOT][2]      
        #self.col[SIDE_IDX_BOT][2][0]    = self.col[SIDE_IDX_LEFT][2][0]
        #self.col[SIDE_IDX_BOT][2][1]    = self.col[SIDE_IDX_LEFT][2][1]
        #self.col[SIDE_IDX_BOT][2][2]    = self.col[SIDE_IDX_LEFT][2][2]
        self.col[SIDE_IDX_BOT][2]       = self.col[SIDE_IDX_LEFT][2]
        #self.col[SIDE_IDX_LEFT][2][0]   = mem_side_top[0]
        #self.col[SIDE_IDX_LEFT][2][1]   = mem_side_top[1]
        #self.col[SIDE_IDX_LEFT][2][2]   = mem_side_top[2]
        self.col[SIDE_IDX_LEFT][2]      = mem_side_top
    elif side_idx == SIDE_IDX_BACK:
      mem_side_bot = self.col[SIDE_IDX_BOT][0].copy()
      if rotate_dir:    #5  
        #self.col[SIDE_IDX_BOT][0][2]    = self.col[SIDE_IDX_LEFT][0][2]
        #self.col[SIDE_IDX_BOT][0][1]    = self.col[SIDE_IDX_LEFT][0][1]
        #self.col[SIDE_IDX_BOT][0][0]    = self.col[SIDE_IDX_LEFT][0][0]
        self.col[SIDE_IDX_BOT][0]       = self.col[SIDE_IDX_LEFT][0]
        #self.col[SIDE_IDX_LEFT][0][2]   = self.col[SIDE_IDX_TOP][0][2]
        #self.col[SIDE_IDX_LEFT][0][1]   = self.col[SIDE_IDX_TOP][0][1]
        #self.col[SIDE_IDX_LEFT][0][0]   = self.col[SIDE_IDX_TOP][0][0]
        self.col[SIDE_IDX_LEFT][0]      = self.col[SIDE_IDX_TOP][0]
        #self.col[SIDE_IDX_TOP][0][2]    = self.col[SIDE_IDX_RIGHT][0][2]
        #self.col[SIDE_IDX_TOP][0][1]    = self.col[SIDE_IDX_RIGHT][0][1]
        #self.col[SIDE_IDX_TOP][0][0]    = self.col[SIDE_IDX_RIGHT][0][0]
        self.col[SIDE_IDX_TOP][0]       = self.col[SIDE_IDX_RIGHT][0]
        #self.col[SIDE_IDX_RIGHT][0][2]  = mem_side_bot[2]
        #self.col[SIDE_IDX_RIGHT][0][1]  = mem_side_bot[1]
        #self.col[SIDE_IDX_RIGHT][0][0]  = mem_side_bot[0]
        self.col[SIDE_IDX_RIGHT][0]     = mem_side_bot
      else:    #11
        #self.col[SIDE_IDX_BOT][0][2]    = self.col[SIDE_IDX_RIGHT][0][2]
        #self.col[SIDE_IDX_BOT][0][1]    = self.col[SIDE_IDX_RIGHT][0][1]
        #self.col[SIDE_IDX_BOT][0][0]    = self.col[SIDE_IDX_RIGHT][0][0]
        self.col[SIDE_IDX_BOT][0]       = self.col[SIDE_IDX_RIGHT][0]
        #self.col[SIDE_IDX_RIGHT][0][2]  = self.col[SIDE_IDX_TOP][0][2]
        #self.col[SIDE_IDX_RIGHT][0][1]  = self.col[SIDE_IDX_TOP][0][1]
        #self.col[SIDE_IDX_RIGHT][0][0]  = self.col[SIDE_IDX_TOP][0][0]
        self.col[SIDE_IDX_RIGHT][0]     = self.col[SIDE_IDX_TOP][0]
        #self.col[SIDE_IDX_TOP][0][2]    = self.col[SIDE_IDX_LEFT][0][2]
        #self.col[SIDE_IDX_TOP][0][1]    = self.col[SIDE_IDX_LEFT][0][1]
        #self.col[SIDE_IDX_TOP][0][0]    = self.col[SIDE_IDX_LEFT][0][0]
        self.col[SIDE_IDX_TOP][0]       = self.col[SIDE_IDX_LEFT][0]
        #self.col[SIDE_IDX_LEFT][0][2]   = mem_side_bot[2]
        #self.col[SIDE_IDX_LEFT][0][1]   = mem_side_bot[1]
        #self.col[SIDE_IDX_LEFT][0][0]   = mem_side_bot[0]
        self.col[SIDE_IDX_LEFT][0]      = mem_side_bot


  def rotate(self, side_idx, rotate_dir=ROT_DIR_CW):
    #print("\nRotate Cube Side")
    
    #rotate the main side, just a array rotatation
    self._rotate_side(side_idx, rotate_dir)

    #rotate the adjacent side
    #first set the sides (upper, lower, left , right)
    #then rotate to unique order, sides must be rotated in a way like the cube is turned 
    # TOP SIDE is correct aligned
    # BOT SIDE --> needs BOT TO TOP 'alignment' means that FRONT and BACK have to be rotated +180/+180)
    # TOP and BOT have same upper/lower, RIGHT and LEFT have same upper/lower
    
    # RIGHT SIDE --> needs RIGHT TO TOP 'alignment' means that FRONT and BACK have to be rotated +270/+90)

    if side_idx == SIDE_IDX_TOP: 
      #print("  Side:      TOP", end='\t')
      upper = self.col[SIDE_IDX_BACK]   #upper facing side
      right = self.col[SIDE_IDX_RIGHT]  #right facing side
      lower = self.col[SIDE_IDX_FRONT]  #lower facing side
      left  = self.col[SIDE_IDX_LEFT]   #left facing side
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
    
    if side_idx == SIDE_IDX_BOT: 
      #print("  Side:      BOT", end='\t')
      upper = self.col[SIDE_IDX_BACK]   #upper facing side
      right = self.col[SIDE_IDX_LEFT]  #right facing side
      lower = self.col[SIDE_IDX_FRONT]  #lower facing side
      left  = self.col[SIDE_IDX_RIGHT]   #left facing side
      upper = np.rot90(upper, 2)
      lower = np.rot90(lower, 2)   
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
      upper = np.rot90(upper, 2)
      lower = np.rot90(lower, 2)   

    if side_idx == SIDE_IDX_RIGHT: 
      #print("  Side:      RIGHT", end='\t')
      upper = self.col[SIDE_IDX_BACK]   #upper facing side
      right = self.col[SIDE_IDX_BOT]    #right facing side
      lower = self.col[SIDE_IDX_FRONT]  #lower facing side
      left  = self.col[SIDE_IDX_TOP]    #left facing side
      upper = np.rot90(upper, 3)
      lower = np.rot90(lower, 1)   
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
      upper = np.rot90(upper, 1)
      lower = np.rot90(lower, 3)   

    if side_idx == SIDE_IDX_LEFT: 
      #print("  Side:      LEFT", end='\t')
      upper = self.col[SIDE_IDX_BACK]   #upper facing side
      right = self.col[SIDE_IDX_TOP]    #right facing side
      lower = self.col[SIDE_IDX_FRONT]  #lower facing side
      left  = self.col[SIDE_IDX_BOT]    #left facing side
      upper = np.rot90(upper, 1)
      lower = np.rot90(lower, 3)   
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
      upper = np.rot90(upper, 3)
      lower = np.rot90(lower, 1)   
  
    if side_idx == SIDE_IDX_FRONT: 
      #print("  Side:      FRONT", end='\t')
      upper = self.col[SIDE_IDX_TOP]   #upper facing side
      right = self.col[SIDE_IDX_RIGHT]    #right facing side
      lower = self.col[SIDE_IDX_BOT]  #lower facing side
      left  = self.col[SIDE_IDX_LEFT]    #left facing side
      lower = np.rot90(lower, 2)
      right = np.rot90(right, 3)
      left = np.rot90(left, 1)   
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
      lower = np.rot90(lower, 2)
      right = np.rot90(right, 1)
      left = np.rot90(left, 3)   


    if side_idx == SIDE_IDX_BACK: 
      #print("  Side:      BACK", end='\t')
      upper = self.col[SIDE_IDX_BOT]   #upper facing side
      right = self.col[SIDE_IDX_RIGHT]    #right facing side
      lower = self.col[SIDE_IDX_TOP]  #lower facing side
      left  = self.col[SIDE_IDX_LEFT]    #left facing side
      upper = np.rot90(upper, 2)
      right = np.rot90(right, 1)
      left = np.rot90(left, 3)   
      self._rotate_adjacent_sides(upper, right, lower, left, rotate_dir)
      upper = np.rot90(upper, 2)
      right = np.rot90(right, 3)
      left = np.rot90(left, 1)   


  
  #turn the whole cube, this changes the color of the middle pin
  #just used for test of rotation actions to better display
  #keep in mind to get bottom side up:  rotate LEFT-TO-TOP twice is not the SAME as FRONT-TO-TOP twice
  #FRONT-to-TOP twice causes 180° rotation of TOP compared to LED-TO-TOP twice
  def turn(self, turn_dir):
    #Left to Top
    if turn_dir==TURN_DIR_LEFT_TO_TOP:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_TOP]   = mem_cube_sides[SIDE_IDX_LEFT]
      cube_sides[SIDE_IDX_LEFT]  = mem_cube_sides[SIDE_IDX_BOT]
      cube_sides[SIDE_IDX_BOT]   = mem_cube_sides[SIDE_IDX_RIGHT]
      cube_sides[SIDE_IDX_RIGHT] = mem_cube_sides[SIDE_IDX_TOP]

      self._rotate_side(SIDE_IDX_FRONT, ROT_DIR_CW)
      self._rotate_side(SIDE_IDX_BACK, ROT_DIR_CCW)
    
    #Right to Top
    if turn_dir==TURN_DIR_RIGHT_TO_TOP:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_TOP]   = mem_cube_sides[SIDE_IDX_RIGHT]
      cube_sides[SIDE_IDX_LEFT]  = mem_cube_sides[SIDE_IDX_TOP]
      cube_sides[SIDE_IDX_BOT]   = mem_cube_sides[SIDE_IDX_LEFT]
      cube_sides[SIDE_IDX_RIGHT] = mem_cube_sides[SIDE_IDX_BOT]

      self._rotate_side(SIDE_IDX_FRONT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BACK, ROT_DIR_CW)

    #axis = x (anchor = LEFT / RIGHT)
    if turn_dir==TURN_DIR_FRONT_TO_TOP:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_TOP]   = mem_cube_sides[SIDE_IDX_FRONT]
      cube_sides[SIDE_IDX_FRONT]  = mem_cube_sides[SIDE_IDX_BOT]
      cube_sides[SIDE_IDX_BOT]   = mem_cube_sides[SIDE_IDX_BACK]
      cube_sides[SIDE_IDX_BACK] = mem_cube_sides[SIDE_IDX_TOP]

      self._rotate_side(SIDE_IDX_LEFT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_RIGHT, ROT_DIR_CW)
      
      self._rotate_side(SIDE_IDX_FRONT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_FRONT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CCW)


    if turn_dir==TURN_DIR_BACK_TO_TOP:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_TOP]   = mem_cube_sides[SIDE_IDX_BACK]
      cube_sides[SIDE_IDX_FRONT]  = mem_cube_sides[SIDE_IDX_TOP]
      cube_sides[SIDE_IDX_BOT]   = mem_cube_sides[SIDE_IDX_FRONT]
      cube_sides[SIDE_IDX_BACK] = mem_cube_sides[SIDE_IDX_BOT]

      self._rotate_side(SIDE_IDX_LEFT, ROT_DIR_CW)
      self._rotate_side(SIDE_IDX_RIGHT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BACK, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BACK, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CCW)

    #axis = z (anchor = TOP / BOT)
    if turn_dir==TURN_DIR_LEFT_TO_FRONT:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_LEFT]   = mem_cube_sides[SIDE_IDX_BACK]
      cube_sides[SIDE_IDX_FRONT]  = mem_cube_sides[SIDE_IDX_LEFT]
      cube_sides[SIDE_IDX_RIGHT]   = mem_cube_sides[SIDE_IDX_FRONT]
      cube_sides[SIDE_IDX_BACK] = mem_cube_sides[SIDE_IDX_RIGHT]
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CW)
      self._rotate_side(SIDE_IDX_TOP, ROT_DIR_CCW)
    
    if turn_dir==TURN_DIR_RIGHT_TO_FRONT:
      cube_sides = self.col
      mem_cube_sides = cube_sides.copy()
      cube_sides[SIDE_IDX_LEFT]   = mem_cube_sides[SIDE_IDX_FRONT]
      cube_sides[SIDE_IDX_FRONT]  = mem_cube_sides[SIDE_IDX_RIGHT]
      cube_sides[SIDE_IDX_RIGHT]   = mem_cube_sides[SIDE_IDX_BACK]
      cube_sides[SIDE_IDX_BACK] = mem_cube_sides[SIDE_IDX_LEFT]
      self._rotate_side(SIDE_IDX_BOT, ROT_DIR_CCW)
      self._rotate_side(SIDE_IDX_TOP, ROT_DIR_CW)

   
  #all possible actions
  def actions(self, action):

    if action==0: self.rotate(SIDE_IDX_TOP, ROT_DIR_CW)
    elif action==6: self.rotate(SIDE_IDX_TOP, ROT_DIR_CCW)
    elif action==1: self.rotate(SIDE_IDX_BOT, ROT_DIR_CW)
    elif action==7: self.rotate(SIDE_IDX_BOT, ROT_DIR_CCW)
    elif action==2: self.rotate(SIDE_IDX_LEFT, ROT_DIR_CW)
    elif action==8: self.rotate(SIDE_IDX_LEFT, ROT_DIR_CCW)
    elif action==3: self.rotate(SIDE_IDX_RIGHT, ROT_DIR_CW)
    elif action==9: self.rotate(SIDE_IDX_RIGHT, ROT_DIR_CCW)
    elif action==4: self.rotate(SIDE_IDX_FRONT, ROT_DIR_CW)
    elif action==10: self.rotate(SIDE_IDX_FRONT, ROT_DIR_CCW)
    elif action==5: self.rotate(SIDE_IDX_BACK, ROT_DIR_CW)
    elif action==11: self.rotate(SIDE_IDX_BACK, ROT_DIR_CCW)
    else: return
    #append 
    self.actions_list.append(action)
 
  #simple actions, less math, less if/else
  def actions_simple(self, action):
    if action==0:         self.rotate_simple(SIDE_IDX_TOP, ROT_DIR_CW)
    elif action==6:       self.rotate_simple(SIDE_IDX_TOP, ROT_DIR_CCW)
    elif action==1:       self.rotate_simple(SIDE_IDX_BOT, ROT_DIR_CW)
    elif action==7:       self.rotate_simple(SIDE_IDX_BOT, ROT_DIR_CCW)
    elif action==2:       self.rotate_simple(SIDE_IDX_LEFT, ROT_DIR_CW)
    elif action==8:       self.rotate_simple(SIDE_IDX_LEFT, ROT_DIR_CCW)
    elif action==3:       self.rotate_simple(SIDE_IDX_RIGHT, ROT_DIR_CW)
    elif action==9:       self.rotate_simple(SIDE_IDX_RIGHT, ROT_DIR_CCW)
    elif action==4:       self.rotate_simple(SIDE_IDX_FRONT, ROT_DIR_CW)
    elif action==10:      self.rotate_simple(SIDE_IDX_FRONT, ROT_DIR_CCW)
    elif action==5:       self.rotate_simple(SIDE_IDX_BACK, ROT_DIR_CW)
    elif action==11:      self.rotate_simple(SIDE_IDX_BACK, ROT_DIR_CCW)
    else: return
    #append 
    self.actions_list.append(action)


  #helper for finding the conjugate action (TOP/CW --> TOP/CCW)
  def conj_action(self, action):
    if action < 6:
      return (action + 6)
    if action >= 6 and action < self.num_actions():
      return (action - 6)
    
  #returns the conjugate actions list
  def get_conj_action_list(self):
    conj_actions_list = []
    for action in self.actions_list:
      conj_actions_list.append(self.conj_action(action))
    return (conj_actions_list)

  #returns the total number of applicable actions
  def num_actions(self):
    return(12)

  #empties the action list
  def clear_action_list(self):
    self.actions_list.clear()
  
  #get a copy from the action list
  def get_action_list(self):
    return self.actions_list.copy()
  
  #rotates a edge on white side 
  #side: Front, back, left, right
  def rot_white_edge(self, side_idx, repetitions):
    degree=10   #degree means how many repetitions it takes to reach the original state again, just information

    #sequence taken from ruwix.com
    #difference to our notation D...down (=bot)   U...up(Top)   
    sequence = {
      SIDE_IDX_FRONT:[ 4, 6, 3, 0],          #F  U' R  U   rotate white edge on front
      SIDE_IDX_LEFT :[ 2, 6, 4, 0],          #L  U' F  U   rotate white edge on left
      SIDE_IDX_BACK :[ 5, 6, 2, 0],          #B  U' L  U   rotate white edge on back
      SIDE_IDX_RIGHT:[ 3, 6, 5, 0],          #R  U' B  U   rotate white edge on right
      }
    for _ in range(repetitions):
      for action in sequence[side_idx]:
        self.actions_simple(action)
  
  def rot_white_corner(self, side_idx, repetitions):
    degree=6   #degree means how many repetitions it takes to reach the original state again, just information

    #sequence taken from ruwix.com
    sequence = {
      SIDE_IDX_FRONT:[10, 7, 4, 1],          #F' D' F  D   rotate white corner on front
      SIDE_IDX_LEFT :[ 8, 7, 2, 1],          #L' D' L  D   rotate white corner on left
      SIDE_IDX_BACK :[11, 7, 5, 1],          #B' D' B  D   rotate white corner on back
      SIDE_IDX_RIGHT:[ 9, 7, 3, 1],          #R' D' R  D   rotate white corner on right
      }
    for _ in range(repetitions):
      for action in sequence[side_idx]:
        self.actions_simple(action)
  
  """
  """
  def swap_second_layer_edge(self, side_idx, direction, repetitions):
    """
    sequence taken from ruwix.com
    difference to our notation D...down (=bot)   U...up(Top)   
    there are 2 basic algos, left and right. they repeat for each side
    we can either use left or right for each 4 sides, no need to use both
    in ruwix algo the cube is upside-down (yellow at up side)
    notation is the same so we replace U with D 
      but order is different R > F --> R > B

    RIGHT-ALGO ORIGINAL
    U  R  U' R' U' F' U  F --> (cube is upside/down, white is at bottom)
    D  R  D' R' D' B' D  B     (translation - white is at top)
      
    LEFT-ALGO ORIGINAL
    U' L' U  L  U  F  U' F' --> (cube is upside/down, white is at bottom)
    D' L' D  L  D  B  D' B'    (translation - white is at top)
    """

    #because algo is for upside down cube its order is changed here with direction
    if direction=="right":
      #left algo
      sequence = {
        SIDE_IDX_FRONT: [ 7, 9, 1, 3, 1, 4, 7, 10 ],    #D' R' D  R  D  F  D' F'         
        SIDE_IDX_LEFT:  [ 7,10, 1, 4, 1, 2, 7, 8  ],    #D' F' D  F  D  L  D' L'      
        SIDE_IDX_BACK:  [ 7, 8, 1, 2, 1, 5, 7, 11 ],    #D' L' D  L  D  B  D' B'      
        SIDE_IDX_RIGHT: [ 7,11, 1, 5, 1, 3, 7, 9  ],    #D' B' D  B  D  R  D' R'      
        }

    elif direction=="left":
      #right algo
      sequence = {
        SIDE_IDX_FRONT: [ 1, 2, 7, 8, 7,10, 1, 4  ],    #D  L  D' L' D' F' D  F          
        SIDE_IDX_LEFT:  [ 1, 5, 7,11, 7, 8, 1, 2  ],    #D  B  D' B' D' L' D  L          
        SIDE_IDX_BACK:  [ 1, 3, 7, 9, 7,11, 1, 5  ],    #D  R  D' R' D' B' D  B      
        SIDE_IDX_RIGHT: [ 1, 4, 7,10, 7, 9, 1, 3  ],    #D  F  D' F' D' R' D  R        
        }
    else:
      return

    for _ in range(repetitions):
      for action in sequence[side_idx]:
        self.actions_simple(action)
  """
  position yellow edges (create yellow cross)
  apply on "dot" , "L" or "line" pattern (3x, 2x, 1x)
  turn the cube between each call, so we use two algos
  """
  def position_yellow_edge(self, algo_num):
    #F  R  U  R' U' F'  original algo
    #F  L  D  L' D' F'  (translation white is at top)
    #B  R  D  R' D' B'  (translation white is at top, cube turned 180° L <-> R)
    sequence = {
      0:  [ 4, 2, 1, 8, 7,10 ],           #F  L  D  L' D' F'       
      1:  [ 5, 3, 1, 9, 7,11 ],           #B  R  D  R' D' B'       
      2:  [ 2, 5, 1,11, 7,8  ],           #L  B  D  B' D' L'       
      3:  [ 3, 4, 1,10, 7,9  ],           #R  F  D  F' D' R'       

      }
    for action in sequence[algo_num]:
      self.actions_simple(action)
  
  """
  swap 2 yellow edges (solve last layer edges)
  apply algo twice to swap opposite edges, for this use combination algo 0+1 or algo 2+3
  """
  def swap_yellow_edge(self, algo_num):
    #R  U  R' U  R  U2  R'  U   original algo
    #R  D  R' D  R  D2  R'  D  (translation white is at top)
    sequence = {
      0:  [ 2, 1, 8, 1, 2, 1, 1, 8, 1 ],          #L  D  L' D  L  D2  L'  D      e10-e11 
      1:  [ 3, 1, 9, 1, 3, 1, 1, 9, 1 ],          #R  D  R' D  R  D2  R'  D      e8-e9 
      2:  [ 4, 1,10, 1, 4, 1, 1,10, 1 ],          #F  D  F' D  F  D2  F'  D      e8-e11 
      3:  [ 5, 1,11, 1, 5, 1, 1,11, 1 ]           #B  D  B' D  B  D2  B'  D      e9-e10 
      }
    for action in sequence[algo_num]:
      self.actions_simple(action)

  def position_yellow_corner(self, algo_num):
    #U  R  U' L' U  R' U' L   original algo
    #D  R  D' L' D  R' D' L  (translation white is at top)
    sequence = {
      0:  [ 1, 3, 7, 8, 1, 9, 7, 2 ],          #D  R  D' L' D  R' D' L    #c4 unchanged
      1:  [ 1, 5, 7,10, 1,11, 7, 4 ],          #D  B  D' F' D  B' D' F    #c5 unchanged
      2:  [ 1, 2, 7, 9, 1, 8, 7, 3 ],          #D  L  D' R' D  L' D' R    #c6 unchanged
      3:  [ 1, 4, 7,11, 1,10, 7, 5 ],          #D  F  D' B' D  F' D' B    #c7 unchanged

      }
    for action in sequence[algo_num]:
      self.actions_simple(action)

  """
  basic algo for orienting last layer corners
    1.  rotate D until a wrong oriented corner is in position
    2.  apply algo x2 or x4 times on the wrong corner, this will bring yellow sticker in position.
    3.  if all corners correct --> Cube finished
        else goto 1:
  
  you need only one of this algos as you can solve the cube with loop describe above.
  suggestion is to search the first misplaced corner and then loop with the algo for that corner over the whole cube
  this may save one move :)
  """
  def rotate_yellow_corner(self, algo_num, repetitions):
    #R' D' R  D    original algo
    #R' U' R  U   (translation white is at top)
    sequence = {
      0:  [ 9, 6, 3, 0  ],          #R' U' R  U   rotate c4
      1:  [11, 6, 5, 0  ],          #B' U' B  U   rotate c5
      2:  [ 8, 6, 2, 0  ],          #L' U' L  U   rotate c6
      3:  [10, 6, 4, 0  ],          #F' U' F  U   rotate c7
      }
    for _ in range(repetitions):
      for action in sequence[algo_num]:
        self.actions_simple(action)


  #save to .JSON file
  def save_to_file(self, filename):
    #open the file, if exists replaces all content
    with open(filename, 'w', encoding='utf-8') as outfile:
      #define data to write
      outdata = {
        'col'     : self.col.tolist(), #np array must be converted to list
        'actions' : self.actions_list,
        'actions_dict' : action_dict,
        'color_dict'   : color_dict,
        'side_dict'    : side_dict,
        }
      json.dump(outdata, outfile, separators=(',', ':'), sort_keys=False, indent=4)    
    return(0)
  

  def load_from_file(self, filename):
    #exists = os.path.isfile(filename)
    with open(filename, 'r', encoding='utf-8') as infile:
      indata = json.loads(infile.read())
      self.col            = np.array(indata['col'])
      self.actions_list   = indata['actions']
    return(0)

    #compares datafield
  def equals(self, cube):
    equal = np.array_equal(cube.col, self.col)
    #print("Equal: " + str(equal))
    return(equal)

  #get score value - each correct color per side is score +1
  #correct means identical with center color
  def score(self):
    center_block  = 0
    score = 0
    #.count works only for rows, therefore step all rows.
    #per row call .count for each color_idx
    for side in self.col:
      center_block = side[1][1]
      for row in side:
        score += row.tolist().count(center_block)   
    return(score)
  
  #compares element per element, if cvalled with ORIG_CUB Data it's result is equal to score()
  def compare(self, cube):
    score = 0
    for i in range(6):
      for j in range(3):
        for k in range(3):
          if self.col[i][j][k] == cube.col[i][j][k]: score += 1
    return(score)
  
  """
  search for <edge_to_search> and return their positition from 0 to 11, 
    search by array value 
    location = cube.search_edge(orig_edge) returns for example:
    [0,1,3,2...]  e00 and e01..correct, e02 is at position e03 and vice versa
      location = cube.search_edge(orig_edge[2:3]) will give you 3 in this case
      location = cube.search_edge(orig_edge[3:4]) will give you 2 in this case
  edge_to_search can be single or multiple edges
  will ignore wrong rotation
  """
  def search_edge(self, edge_to_search):
    #get actual edges from cube
    edges = self.get_edge(sort=True)
    edge_location = []
    cnt = 0
    if type(edge_to_search[0]) is not list:
      edge_to_search = [edge_to_search]

    for edge_target in edge_to_search:
      #sort each target
      edge_target = np.sort(edge_target).tolist()
      for n in range(12): 
        if edges[n] == edge_target:
          #print("Edge to solve e%02d found at location: %d" % (cnt, n))
          cnt += 1
          edge_location.append(n)
    #only one element -> output as number, otherwise output as list
    if len(edge_location)==1:       return edge_location[0]
    else:                           return edge_location

  """
  search for <corner_to_search> and return their positition from 0 to 7, 
    search by array value 
    location = cube.search_corner(orig_corner) returns for example:
    [0,1,3,2...]  c00 and c01..correct, c02 is at position c03 and vice versa
      location = cube.search_corner(orig_corner[2:3]) will give you 3 in this case
      location = cube.search_corner(orig_corner[3:4]) will give you 2 in this case
  corner_to_search can be single or multiple edges
  will ignore wrong rotation
  """
  def search_corner(self, corner_to_search):
    #get actual edges from cube
    corners = self.get_corner(sort=True)
    corner_location = []
    cnt = 0
    if type(corner_to_search[0]) is not list:
      corner_to_search = [corner_to_search]

    for corner_target in corner_to_search:
      #sort each target
      corner_target = np.sort(corner_target).tolist()
      for n in range(8): 
        if corners[n] == corner_target:
          #print("Edge to solve e%02d found at location: %d" % (cnt, n))
          cnt += 1
          corner_location.append(n)
    #only one element -> output as number, otherwise output as list
    if len(corner_location)==1:     return corner_location[0]
    else:                           return corner_location



  """
  edges are numbered from up-mid-down, used in 6-step standard solving sequence
  get all edges or a single edge location given by <location> from 0 to 11
  optional sorting color values, <sort>=True
  """
  def get_edge(self, location=None, sort=False):
    edge_block       = [[0]*2 for _ in range(12)]
    edge_block[0][0] = self.col[SIDE_IDX_TOP][0][1]
    edge_block[0][1] = self.col[SIDE_IDX_BACK][2][1]
    
    edge_block[1][0] = self.col[SIDE_IDX_TOP][1][2]
    edge_block[1][1] = self.col[SIDE_IDX_RIGHT][1][0]
    
    edge_block[2][0] = self.col[SIDE_IDX_TOP][2][1]
    edge_block[2][1] = self.col[SIDE_IDX_FRONT][0][1]
    
    edge_block[3][0] = self.col[SIDE_IDX_TOP][1][0]
    edge_block[3][1] = self.col[SIDE_IDX_LEFT][1][2]

    edge_block[4][0] = self.col[SIDE_IDX_BACK][1][2]
    edge_block[4][1] = self.col[SIDE_IDX_RIGHT][0][1]
    
    edge_block[5][0] = self.col[SIDE_IDX_RIGHT][2][1]
    edge_block[5][1] = self.col[SIDE_IDX_FRONT][1][2]

    edge_block[6][0] = self.col[SIDE_IDX_FRONT][1][0] 
    edge_block[6][1] = self.col[SIDE_IDX_LEFT][2][1]
    
    edge_block[7][0] = self.col[SIDE_IDX_LEFT][0][1]
    edge_block[7][1] = self.col[SIDE_IDX_BACK][1][0]

    edge_block[8][0] = self.col[SIDE_IDX_BOT][0][1]
    edge_block[8][1] = self.col[SIDE_IDX_BACK][0][1]
    
    edge_block[9][0] = self.col[SIDE_IDX_BOT][1][2]
    edge_block[9][1] = self.col[SIDE_IDX_LEFT][1][0]
    
    edge_block[10][0] = self.col[SIDE_IDX_BOT][2][1]
    edge_block[10][1] = self.col[SIDE_IDX_FRONT][2][1]
    
    edge_block[11][0] = self.col[SIDE_IDX_BOT][1][0]
    edge_block[11][1] = self.col[SIDE_IDX_RIGHT][1][2]
    #requested sorted values
    if sort == True:
      for n in range(12):
        edge_block[n]      = np.sort(edge_block[n]).tolist()
    #requested a specific location
    if location: return edge_block[location]

    return edge_block

  """
  corners are numbered starting from up/left position in clockwise direction
  continuing on down/left in clockwise direction
  get all corners or a single corner location given by <location> from 0 to 7
  optional sorting color values, <sort>=True
  """
  def get_corner(self, location=None, sort=False):
    corner_block  = [[0]*3 for _ in range(8)]  
    corner_block_idx = self.N_DIM - 1

    corner_block[0][0] = self.col[SIDE_IDX_TOP][0][0]
    corner_block[0][1] = self.col[SIDE_IDX_BACK][corner_block_idx][0]
    corner_block[0][2] = self.col[SIDE_IDX_LEFT][0][corner_block_idx]

    corner_block[1][0] = self.col[SIDE_IDX_TOP][0][corner_block_idx]
    corner_block[1][1] = self.col[SIDE_IDX_BACK][corner_block_idx][corner_block_idx]
    corner_block[1][2] = self.col[SIDE_IDX_RIGHT][0][0]

    corner_block[2][0] = self.col[SIDE_IDX_TOP][corner_block_idx][corner_block_idx]
    corner_block[2][1] = self.col[SIDE_IDX_FRONT][0][corner_block_idx]
    corner_block[2][2] = self.col[SIDE_IDX_RIGHT][corner_block_idx][0]

    corner_block[3][0] = self.col[SIDE_IDX_TOP][corner_block_idx][0]
    corner_block[3][1] = self.col[SIDE_IDX_FRONT][0][0]
    corner_block[3][2] = self.col[SIDE_IDX_LEFT][corner_block_idx][corner_block_idx]

    corner_block[4][0] = self.col[SIDE_IDX_BOT][0][0]
    corner_block[4][1] = self.col[SIDE_IDX_BACK][0][corner_block_idx]
    corner_block[4][2] = self.col[SIDE_IDX_RIGHT][0][corner_block_idx]

    corner_block[5][0] = self.col[SIDE_IDX_BOT][0][corner_block_idx]
    corner_block[5][1] = self.col[SIDE_IDX_BACK][0][0]
    corner_block[5][2] = self.col[SIDE_IDX_LEFT][0][0]

    corner_block[6][0] = self.col[SIDE_IDX_BOT][corner_block_idx][corner_block_idx]
    corner_block[6][1] = self.col[SIDE_IDX_FRONT][corner_block_idx][0]
    corner_block[6][2] = self.col[SIDE_IDX_LEFT][corner_block_idx][0]

    corner_block[7][0] = self.col[SIDE_IDX_BOT][corner_block_idx][0]
    corner_block[7][1] = self.col[SIDE_IDX_FRONT][corner_block_idx][corner_block_idx]
    corner_block[7][2] = self.col[SIDE_IDX_RIGHT][corner_block_idx][corner_block_idx]
    #requested sorted values
    if sort == True:
      for n in range(8):
        corner_block[n]      = np.sort(corner_block[n]).tolist()
    #requested a specific location
    if location: return corner_block[location]
    return corner_block



  def get_center(self, location=None, sort=False):
    center_block  = [0] * 6
    for i in range(6):
      center_block[i] = self.col[i][1][1]
    #requested sorted values
    if sort == True:
        center_block      = np.sort(corner_block).tolist()
    #requested a specific location
    if location: return center_block[location]
    return center_block    



  def self_test(self):
    #print("Check consistency of data")
    #load color data for corner, edge and center blocks
    #edge has 3 visible sides, corner has 2 visible sides

    #this is sorted validation data, that is derived from the original cube
    #it hast to be identical with sorted data from any cube
    orig_cube           = cube.tRubikCube()
    edge_block_valid    = orig_cube.get_edge(sort=True)
    corner_block_valid  = orig_cube.get_corner(sort=True)
    center_block_valid  = orig_cube.get_center(sort=True)
    
    #sorted data from actual cube
    edge_block = self.get_edge(sort=True)
    center_block = self.get_center(sort=True)
    corner_block = self.get_corner(sort=True)

    corner_block_test   = [False]*8
    edge_block_test     = [False]*12
    center_block_test   = [False]*6

    #check all center, edge and corner blocks
    #more simple would be just compare the whole arrays
    
    #each color combination from validation data has to be present once
    for corner in corner_block:
      for i in range(len(corner_block_valid)):
        if corner == corner_block_valid[i]: 
          corner_block_test[i] = True
          break
    #print(corner_block_test)   
    if corner_block_test.count(True) != len(corner_block_test):
      return(False)

    for edge in edge_block:
      for i in range(len(edge_block_valid)):
        if edge == edge_block_valid[i]: 
          edge_block_test[i] = True
          break
    #print(edge_block_test)   
    if edge_block_test.count(True) != len(edge_block_test):
      return(False)

    for center in center_block:
      for i in range(len(center_block_valid)):
        if center == center_block_valid[i]: 
          center_block_test[i] = True
          break
    #print(center_block_test)   
    if center_block_test.count(True) != len(center_block_test):
      return(False) 
    
    return(True)
  
def cube_algo_check():  
  Cube = tRubikCube()
  console_clear()
  Cube.print_2d()
  
  Cube = tRubikCube()
  Cube.solve_second_layer_edge(1, "test", 1)
  Cube.print_2d()
  
  Cube = tRubikCube()
  Cube.solve_second_layer_edge(2, "test", 1)
  Cube.print_2d()
  
  input()
  #Cube = tRubikCube()



#Run Selftest on all actions
def cube_actions_check():

  Orig_Cube = tRubikCube()
  num_actions = Orig_Cube.num_actions()

  #perform each action once from original state, and run selftest afterwards
  print("check single actions (std vs. simple)")
  for i in range(num_actions):
    Cube_std    = tRubikCube()
    Cube_simple = tRubikCube()
    Cube_std.actions(i)
    Cube_simple.actions_simple(i)
    result  = Cube_std.self_test()
    result2 = Cube_simple.self_test()
    equals  = Cube_std.equals(Cube_simple)
    print("Equals = " + str(equals) + "\tAction:  " + str(i) + "\tCube Self-Test Result: "+ str(result) + " / " + str(result2))
   
  
  print("\nCheck random actions (std vs. simple)")
  random.seed()
  iter_steps = 12000
  print("  Actions: %d, Iter_Steps: %d" %(num_actions, iter_steps))
  print("  ", end="")
  Cube_std = tRubikCube()
  Cube_simple = tRubikCube()
  for i in range(iter_steps):
    action=random.randrange(0, num_actions)
    Cube_std.actions(action)  
    Cube_simple.actions_simple(action)
    result = Cube_simple.equals(Cube_std)            #1.72 -> 1.99
    if i % 2000 == 0: print(".", end="", flush=True)
    if(result is False): 
      print("FALSE, action = %d, iter = %d" % (action, i))
      Cube_std.print_2d()
      Cube_simple.print_2d()
      input()
   
  iCount = []
  [iCount.append(Cube_simple.actions_list.count(i)) for i in range(num_actions)]
  print("\n  Actions occurence: %s" % str(iCount))


def cube_score_check():
  print("check Cube Score-ing Function...")
  Orig_Cube = tRubikCube()
  score = Orig_Cube.score()
  print("Original Cube Score (9x6=54): %d" % score)
  Orig_Cube.print_2d()

  num_actions = Orig_Cube.num_actions()
  Cube_simple = tRubikCube()
  random.seed()
  iter_steps = 6
  for i in range(iter_steps):
    action=random.randrange(0, num_actions)
    Cube_simple.actions_simple(action)
    score = Cube_simple.score()
    compare = Cube_simple.compare(Orig_Cube)
    print("Rotated Cube Score: %02d \t Rotated Cube Compare to Orig Score: %02d" % (score, compare))
    Cube_simple.print_2d()
    input()



def cube_benchmark():
  Orig_Cube = tRubikCube()
  num_actions = Orig_Cube.num_actions()
  iter_steps = 10000
  total_actions = iter_steps * num_actions

  start = datetime.datetime.now()  
  print("\nSimple Actions Benchmark")
  print("  Actions per Step: %d, Iter_Steps: %d, Total-Actions(actions*steps)= %d" %(num_actions, iter_steps, total_actions))
  print("  ", end="")

  for i in range(iter_steps):
    Cube_simple = tRubikCube()                        
    #Cube_simple = deepcopy(Orig_Cube)                     
    if i % 2000 == 0: print(".", end="", flush=True)
    [Cube_simple.actions_simple(0) for j in range(num_actions)]
    #for j in range(num_actions):
      #Cube_simple.actions_simple(j)
    result = Cube_simple.equals(Orig_Cube)              

  stop = datetime.datetime.now()
  delta = stop-start
  delta_seconds = float(delta.total_seconds())
  if delta_seconds > 60:
    print("\n  Total time = %.2fmin" % (delta_seconds/60.0))
  else:
    print("\n  Total time = %.2fsec" % delta_seconds)
  if delta_seconds > 0.0: 
    iter_per_sec = iter_steps / delta_seconds
    print("  Iterations per seconds = %d" % int(iter_per_sec))
  else:
    print("  Iterations per seconds = <not reliable, to short>")
  

  start = datetime.datetime.now()
  print("\nStandard Actions Benchmark")
  print("  Actions per Step: %d, Iter_Steps: %d, Total-Actions(actions*steps)= %d" %(num_actions, iter_steps, total_actions))
  print("  ", end="")
  for i in range(iter_steps):
    Cube_std    = tRubikCube()                         
    #Cube_std = deepcopy(Orig_Cube)                     
    if i % 2000 == 0: print(".", end="", flush=True)
    [Cube_std.actions(0) for j in range(num_actions)]
    #for j in range(num_actions):
      #Cube_std.actions(0)
    result = Cube_std.equals(Orig_Cube)              

  stop = datetime.datetime.now()
  delta = stop-start
  delta_seconds = float(delta.total_seconds())
  if delta_seconds > 60:
    print("\n  Total time = %.2fmin" % (delta_seconds/60.0))
  else:
    print("\n  Total time = %.2fsec" % delta_seconds)
  if delta_seconds > 0.0: 
    iter_per_sec = iter_steps / delta_seconds
    print("  Iterations per seconds = %d" % int(iter_per_sec))
  else:
    print("  Iterations per seconds = <not reliable, to short>")


def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=60')  #12*4 +1
  cube_algo_check()
  #cube_actions_check()
  #cube_score_check()
  #cube_benchmark()

  

if __name__=="__main__":
  main()
