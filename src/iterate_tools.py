#CHANGELOG: iterate_tools.py
#AUTHOR: SL

#04.01.2020 
#added methodes to tIterate:
# - def generator(self):  - is faster then next() from initial version, read notes
# - def set_step(self, iter_step):
# - def get_step(self):
# - def save_to_file(self, filename):
# - def _load_from_file(self, filename):
#modified tIterate constructor:
# - def __init__(self, depth=1, num_actions=12, start=0, filename=None): 
#     supports now reading constructor values from a file

#27.12.2019
#initial version 

import os
import datetime
import json

from helpers import console_clear, num_to_str_si




class tIterate:
  #inits for given depth and actions
  #len of sequence is depth
  #num_actions means the number of variations
  #example: numactions = 2, depth = 3 --> sequences: [0,0,0] [1,0,0] [0,1,0] [1,1,0] [0,0,1] [1,0,1] [0,1,1] [1,1,1]
  #total num = 2**3 = 8, equals to len(all_sequences)
  def __init__(self, depth=1, num_actions=12, start=0, filename=None):
    #filename given - read from file
    if filename:
      self._load_from_file(filename)
    #otherwise use data from constructor
    else:
      self.depth            = depth         #iteration depth
      self.num_actions      = num_actions   #
      self.iter_action_list = [0] * depth   #actual state of iteration, init to zero
      self.set_step(start)                  #set iteration starting value
    
    self.last_action = self.num_actions -1 
    #calculate total number of iterations, no exp. function for int so use a loop
    self.total_num_iter = 1
    #exponentiation
    for exp in range(self.depth):
      self.total_num_iter *= self.num_actions

    
  #getter for total number of iterations
  def get_total_num(self):
    return (self.total_num_iter)


  #will return consecutive iteration sequences starting with [0,0,...], then [1,0,..] then [2,0,...]
  #will deliver all possible states including repeated elements 
  def next(self):
    #actual state of self.iter_action_list is retval
    retval = self.iter_action_list.copy()
    #optional revert the sequence output
    #retval.reverse()   
    #last_action = self.num_actions-1
    #blcarry FLAG, marks that next digit needs to be incremented, if we do not get a carry-flag we can break the loop immediate as there will be no further change
    #similar to CARRY Flag in MCU when doing ADD
    blcarry = True
    for i in range(self.depth):
      if blcarry:
        if self.iter_action_list[i] < self.last_action:
          self.iter_action_list[i] = self.iter_action_list[i] + 1
          break
        else:
          self.iter_action_list[i] = 0
          blcarry = True
    return (retval)

  #will do the same as next() but with yield, is quite faster as no copy() is needed
  #yield returns the value and leaves the function
  #the difference to return() is that the the next function-entry does not start at beginning of the generator func
  #it continues from the position of yield.
  #!!!attention when using yield, as the generator-function pointer state cannot be reset!!!
  def generator(self):
    while True:
      yield self.iter_action_list
      blcarry = True
      for i in range(self.depth):
        if blcarry:
          if self.iter_action_list[i] < self.last_action:
            self.iter_action_list[i] += 1
            break
          else:
            self.iter_action_list[i] = 0
            blcarry = True
  
  
  #set self.iter_action_list[i] to iter_step number
  #when working with generator it is necessary to re-instantiate the iterator object to get the correct starting value
  #if the generator has been called once and then set_step() is called the generator will yield start+1. 
  #set_step(1) for the first time before generator was called
    #yield1 [1, 0, 0]
    #yield2 [2, 0, 0]
  #set_step(1) - generator func is not reset
    #yield1 [2, 0, 0]  - yields iter_step(start+1)
  #reset the iter_step to zero when generator was called once we have to give them total_num_iter.
  #as the generator is rotating it will carry the overflow flag and start again from zero
  #best practice would be to re-instantiate the tIterator object very time and give them the starting value in the constructor
  def set_step(self, iter_step):
    for i in range(self.depth): 
      self.iter_action_list[i] = iter_step % self.num_actions
      iter_step = int(iter_step / self.num_actions)
    return self.iter_action_list
  
  #calculate iteration number from self.iter_action_list
  def get_step(self):
    multiplier = 1
    iter_step = 0
    for i in range(self.depth): 
      iter_step += self.iter_action_list[i] * multiplier
      multiplier *= self.num_actions
    return iter_step 

  #save Iterator Data to File
  def save_to_file(self, filename):
    #open the file, if exists replaces all content
    with open(filename, 'w', encoding='utf-8') as outfile:
      #define data to write
      iter_step   = self.get_step()
      iter_steps  = self.get_total_num()

      str_iteration = "%s/%s" % (num_to_str_si(iter_step), num_to_str_si(iter_steps))
      str_progress  = "%.3f%%" % (float(iter_step / iter_steps * 100.0))

      outdata = {
        'depth'         : self.depth, 
        'num_actions'   : self.num_actions,
        'step'          : self.get_step(),
        'iteration'     : str_iteration,
        'progress'      : str_progress
        }
      json.dump(outdata, outfile, separators=(',', ':'), sort_keys=False, indent=4)    

  #load iterate data from file
  def _load_from_file(self, filename):
    #exists = os.path.isfile(filename)
    with open(filename, 'r', encoding='utf-8') as infile:
      indata = json.loads(infile.read())
      self.depth            = indata['depth']
      self.num_actions      = indata['num_actions']
      step                  = indata['step']
      self.iter_action_list = [0] * self.depth    #actual state of iteration, init to zero
      self.set_step(step)                         #set iteration starting step


def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=160 lines=60')  #12*4 +1
  

  max_depth = 3
  num_actions = 4
  
  test_iterator = tIterate(max_depth, num_actions, 0)
  iter_steps    = test_iterator.get_total_num()
  print("\nIteration Depth:   %d" %max_depth, end= "")
  print("  Num_Actions:       %d" %num_actions, end="")
  print("  Num_Iterations:    %d" %iter_steps)
  
  for i in range (32):
    test_iterator = tIterate(max_depth, num_actions, i)
    iter_steps    = test_iterator.get_total_num()
    iter_step     = test_iterator.get_step()
    iter_func     = test_iterator.generator()
    iter_sequence = next(iter_func)
    print("Iteration Number:   %03d   " % iter_step, end = "")
    print("Iteration Sequence: %s     " % str(iter_sequence))
  
  
    
  max_depth = 3
  num_actions = 4  
  starting_value = 2
  print("\n---------Test Iterator------------")
  print("---------Save to File-------------")
  test_iterator = tIterate(max_depth, num_actions, starting_value)
  iter_steps    = test_iterator.get_total_num()  
  iter_step     = test_iterator.get_step()
  iter_func     = test_iterator.generator()
  iter_sequence = next(iter_func)
  print("Iteration Depth:     %d" %test_iterator.depth, end= "")
  print("  Num_Actions:       %d" %test_iterator.num_actions, end="")
  print("  Num_Iterations:    %d" %iter_steps)
  print("Iteration Number:   %03d   " % iter_step, end = "")
  print("Iteration Sequence: %s     " % str(iter_sequence))
  test_iterator.save_to_file("test_iterator.json")
  


  print("\n---------Test Iterator2-----------")
  print("---------Load from File-----------")
  test_iterator2 = tIterate(filename="test_iterator.json")
  iter_steps     = test_iterator2.get_total_num()
  iter_step      = test_iterator2.get_step()
  iter_func      = test_iterator2.generator()
  iter_sequence  = next(iter_func)
  print("Iteration Depth:     %d" %test_iterator2.depth, end= "")
  print("  Num_Actions:       %d" %test_iterator2.num_actions, end="")
  print("  Num_Iterations:    %d" %iter_steps)
  print("Iteration Number:   %03d   " % iter_step, end = "")
  print("Iteration Sequence: %s     " % str(iter_sequence))



  print("Press Enter to continue")
  input()
  max_depth = 8
  num_actions = 12
  starting_value = 0
  solutions = []
  iter_per_sec = 2000 #approx
  for itv_deep in range(1,max_depth):
  #for itv_deep in range(2,3):
    if len(solutions) > 0: break
    seq_iterator = tIterate(itv_deep, num_actions, starting_value)
    iter_steps   = seq_iterator.get_total_num()
    iter_func    = seq_iterator.generator()

    print("\nIteration Depth:  "+str(itv_deep), end='')
    print("  Num_Iterations: "+str(iter_steps))
    #print("  Initial Iter:   "+str(Orig_Cube.next_iteration()))
    start = datetime.datetime.now()
    time_last = datetime.datetime.now()
    iter_last = 0
   

    for iter in range(iter_steps):
      #iteration_step = seq_iterator.next()
      iteration_step = next(iter_func)  
      #print(str(iteration_step))
      #if (iter % (int(iter_steps/100)+1) == 0): 
      itercnt = iter-iter_last
      #if (itercnt >= iter_per_sec) or not (iter % (int(iter_steps/100)+1)):  
      if itercnt >= 500000:
        iter_last = iter
        #print(".", end='', flush=true);
        actual = datetime.datetime.now()
        actual_delta = actual-start
        actual_delta = actual_delta.total_seconds() #miliseconds
        if(actual_delta > 0.0):
          iter_per_sec = iter / actual_delta
          progress = 100.0*iter/iter_steps
          remaining = iter_steps-iter
          time_to_completion = remaining / iter_per_sec
          print("\rprogress=%.2f%%  iter_num=%d  iter_per_sec=%dk  remainig=%.2fsec          " % (progress, iter, int(iter_per_sec/1000.0),time_to_completion ), end='')


    stop = datetime.datetime.now()
    delta = stop-start
    delta_seconds = float(delta.total_seconds())
    print("\nTotal seconds = %.3f" % delta_seconds)
    if delta_seconds < 1.0: delta_seconds = 1.0
    print("Iterations per seconds = %dk" % int(iter_steps / delta_seconds / 1000.0))


    if not solutions:
      print("No Solution found")


if __name__=="__main__":
  main()
