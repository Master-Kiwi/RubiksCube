#CHANGELOG: main_multiproc.py
#AUTHOR: SL

#04.01.2020 
# first version
#   use multiprocessing to search for Rubiks Cube solution
#     dispatch jobs using queue
#     results returned via pipe
#   progress is saved to file (requires iterate_tools.py from 04.01.2020)
#   use PROGRESS_SAVE_ITV to control the interval between saves to disk, default to 60 sec
#   if progress file is present it will resume automatically from that point
#   you can use progress file to set a specific starting point for iteration


import os 
import time
import datetime
from copy import copy, deepcopy
from multiprocessing import Pool, Process, Queue, Pipe, Lock

#self written submodules
from RubiksCube import tRubikCube
from iterate_tools import tIterate
from helpers import console_clear


#worker process for bruteforce iteration
#init arg is filename of RubikCube that should be worked on
#q...queue, is shared via all worker-processes
#l...lock, if an action in the worker is not allowed to be interrupted, for example a print to the console
#conn....pipe connection, this is a 2-way 1-1 connection, from each child to the parent. it's used only in one direction. at the parent it has only one instance (parent_conn)
def worker_process(filename, q, l, conn):
  Cube = tRubikCube()
  
  #-------Load Target Cube----------
  l.acquire()
  #info("Worker-Job")  
  Cube.load_from_file(filename)
  l.release()
  
  Solved_Cubes = []

  while True:
    #get actual job from queue
    job_data = q.get()
    #decode values
    iter_start  = job_data[0]
    itv_deep    = job_data[1]
    batch_size  = job_data[2]

    #kill-flag
    if iter_start == -1: break

    #init iterator with starting value
    seq_iterator = tIterate(itv_deep, 12, iter_start)
    iter_steps   = seq_iterator.get_total_num()
    iter_func    = seq_iterator.generator()
     
    iter_last = 0
    #print("PID: %d  Job data: %s" %(os.getpid(), str(job_data)))
    #compute iteration batch
    for iter in range(batch_size):
      iteration_step = next(iter_func)           #using generator is faster
      Test = tRubikCube()
      #Perform all rotate actions, iteration_step is a list of actions, list comprehension is faster
      [Test.actions_simple(action) for action in iteration_step]
      #check for match with target, we need to do this only after the rotation is complete, as we checked all other variants before
      if Test.equals(Cube): 
        #print("----!!!Solution found: %s!!!-----" % str(iteration_step))
        #Solved_Cubes.append(deepcopy(Test))
        
        #notify manager process, send result, last iteration number and sequence if solution is found
        conn.send([True, iter_start+iter, iteration_step])
 
    conn.send([False, iter_start+batch_size, []])

  #print('Exiting worker-function')
  #conn.close()


  
  
    
def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=120 lines=40')  #12*4 +1

  Cube = tRubikCube()
  
  #------load a prerotated cube-----------
  #filename = "../data/simple_cube_04.json"
  #filename = "../data/moderate_cube_08.json"
  #filename = "../data/complex_cube_12.json"
  filename = "../data/real_cube.json"

  script_dir      = os.path.dirname(__file__) #<-- absolute dir the script is in
  cube_file_path   = os.path.join(script_dir, filename)
  Cube.load_from_file(cube_file_path)  

  #progress is always stored in the same file
  progress_filename     = ("%s_progress.json" % filename)
  progress_file_path    = os.path.join(script_dir, progress_filename)
  
  
  PROGRESS_SAVE_ITV = 60    #seconds

  Cube.load_from_file(cube_file_path)
  Cube.print_2d()
  result = Cube.self_test()
  print("Cube Self Test Result: "+ str(result))
  if result == False: exit(0)

  #no solution in cube.file
  if Cube.actions_list: 
    test_solution= []   
    print("No Sequence in Cube-Data found")
  else:
    print("Action sequence:                     " + str(Cube.actions_list))
    #calculate the solution sequence that should be found
    test_solution = Cube.get_conj_action_list()
    test_solution.reverse()
    print("Conjugate action sequence (solution):" + str(test_solution))
  
  #check for progress file
  exists = os.path.isfile(progress_file_path)
  if exists:
    print("Found a Progress File: '%s'- Progress Statistics: " % progress_filename)
    resume_iterator = tIterate(filename = progress_file_path)  
    iter_steps    = resume_iterator.get_total_num()
    iter_step     = resume_iterator.get_step()
    print("Depth:  %03d" %resume_iterator.depth, end= "")
    print("  Num_Actions:  %03d" %resume_iterator.num_actions, end="")
    print("  Iteration:  %d/%d" %(iter_step, iter_steps), end="")
    print("  Progress:  %.3f%%" % float(iter_step / iter_steps * 100.0))
    
    itv_deep        = resume_iterator.depth
    num_actions     = resume_iterator.num_actions 
    iterator_start  = iter_step

  else:
    print("No Progress File, start from beginning")
    itv_deep        = 1
    num_actions     = 12
    iterator_start  = 0
  
  print("Press Enter to Start")
  input()

  print("Search Solution with Iterative Deepening")
  print("Starting Multiple Worker Processes...")

  
  #cpu-count -1 workers, 1 cpu slot left for manager
  num_process = os.cpu_count()-1    
  print("Using %d CPUs"%num_process) 
  q = Queue()
  l = Lock()
  parent_conn, child_conn = Pipe()

  #info('main-job')
  process_list = [Process(target=worker_process, args=(cube_file_path, q, l, child_conn)) for i in range(num_process)]
  [proc.start() for proc in process_list]
  time.sleep(2)
  
  solutions = []
  iter_per_sec = 2000 #approx starting value
  #search solution, increase depth after each run
  while True:
    start = datetime.datetime.now()     #iteration depth starting time
    last_save_time = start              #set last save timeestamp to now
    time.sleep(0.2)                     #guarantee that first time-delta is > 0 (for small iteration batches)
    
    #this iterator is used for saving progress
    save_iterator   = tIterate(itv_deep, num_actions, iterator_start)

    #Iterator and print Info
    seq_iterator = tIterate(itv_deep, num_actions, iterator_start)
    #Iteration steps to compute depend on iterator starting value
    iter_steps   = seq_iterator.get_total_num() - iterator_start 
    time_to_completion = float(iter_steps / iter_per_sec)


    #output info for actual Iteration Depth
    print("\nDepth:  %d" % itv_deep, end='')
    print("  Timestamp: %s" % str(start), end="")
    if iterator_start > 100000:
      print("  Iteration: %dk" % (iterator_start/1000), end='')
    else:
      print("  Iteration: %d" % iterator_start, end='')
    if iter_steps > 100000:
      print("/%dk" % (iter_steps/1000), end='')
    else:
      print("/%d" % iter_steps, end='')
    
    if(time_to_completion > 60): 
      print("  Estimated Time: %.2fmin" % (time_to_completion/60.0) )
    else:
      print("  Estimated Time: %.2fsec" % time_to_completion )

    
    #define initial batch size
    batch_size = 10000
    #very low number of iterations - only one batch
    if(iter_steps < batch_size):  batch_size = iter_steps    
    


    #loop over all batches and queue them to the workers, count the number of queue items
    queued_jobs = 0
    for iter_start in range(iterator_start, iter_steps, batch_size):
      #build worker job starting data
      job_data = [0] * 3
      job_data[0] = int(iter_start)
      job_data[1] = int(itv_deep)
      #truncate last batch, it would be no problem to calculate more as iterator is restarting from beginning
      #but it might yield double results on high batch_size
      if iter_start+batch_size > iter_steps:
        job_data[2] = int(iter_steps - iter_start)
      else: 
        job_data[2] = int(batch_size)
      #send to queue, any of the workers will receive and process
      q.put(job_data, block=True, timeout=None)   #send blocking, if queue is full it will wait until enough space free
      #typical all jobs are queued immediately
      queued_jobs += 1

    print('All batches queued for this iteration (%d)... wait for completion' % queued_jobs)
    
    #Poll the RX-Pipe for answer from the worker-jobs. there must be queued_jobs answers when all batches are processed
    num_solutions=0
    last_iter = 0
    #as long as answers are missing...
    while queued_jobs > 0:
      queued_jobs -= 1                                #decrement queued_jobs
      data_available = parent_conn.poll(timeout=None) #blocking wait for data
      if data_available:
        batch_result = parent_conn.recv()   #blocking, wait for recepetion, will not block here as we checked before with poll()
        result =          batch_result[0]   #True if solution is found
        iter =            batch_result[1]   #last iteration number or actual iteration number if result=True
        iteration_step =  batch_result[2]   #empt list [] or action sequence if result=True

        actual = datetime.datetime.now()

        delta_save_time = actual - last_save_time
        delta_save_time = delta_save_time.total_seconds() 
        #check if progress needs to be saved
        #we save not the actual iteration-number, we save the old iteration number.
        #this is done to prevent from missing iterations, because multiple worker jobs deliver their results unordered
        #the jobs are ordered qued, but the answers will come unordered
        if delta_save_time > float(PROGRESS_SAVE_ITV):
          print("\nSave Progress: %s" %progress_filename, end="")
          print("  Iteration Number: %d" % save_iterator.get_step(), end="")
          print("  Timestamp: %s" % str(last_save_time))
          save_iterator.save_to_file(progress_file_path)     #save to file
          
          #only the iteration step will increment, depth will increment on restart of outer-loop
          save_iterator.set_step(iter)
          last_save_time = actual     #set last save time
        
        #Print Progress Indication
        actual_delta = actual-start
        actual_delta = actual_delta.total_seconds() 
        if(actual_delta > 0.0) and (iter > last_iter):
          last_iter = iter
          iter_per_sec = iter / actual_delta
          progress = 100.0*iter/iter_steps
          remaining = iter_steps-iter
          time_to_completion = remaining / iter_per_sec
          if time_to_completion > 60:
            print("\r[Progress=%.3f%%  iter_num=%dk/%dk  iter_per_sec=%05d  remaining=%.2fmin]          " % (progress, (iter/1000), (iter_steps/1000), iter_per_sec ,(time_to_completion/60) ), end='')
          else:
            print("\r[Progress=%.3f%%  iter_num=%dk/%dk  iter_per_sec=%05d  remaining=%.2fsec]          " % (progress, (iter/1000), (iter_steps/1000), iter_per_sec ,time_to_completion ), end='')
        
        #check 
        if result:
          #perform the action sequence on new cube - this will result in the target cube
          Solved_Cube = tRubikCube()
          Orig_Cube = tRubikCube() 
          for action in iteration_step:
            Solved_Cube.actions_simple(action)
          #conjugate and revert the action list
          test_solution = Solved_Cube.get_conj_action_list()
          test_solution.reverse()

          #reload the target cube from file and optional clear it's action list
          Cube.load_from_file(cube_file_path)  
          
          #Cube.clear_action_list()
          #apply the reverted action list on the target cube, must result in the original cube state
          for action in test_solution:
            Cube.actions_simple(action)
          
          #back-check 
          print("\n  Solution Correct: %s" % str(Cube.equals(Orig_Cube)))
          #store the result in a solution file
          solved_filename = "%s_solution_%02d.json" % (filename, num_solutions)
          solved_cube_file_path   = os.path.join(script_dir, solved_filename)
  
          Cube.save_to_file(solved_cube_file_path)

          solutions.append(test_solution)
          print("  Solution sequence found [%02d]       :%s" % (num_solutions, str(test_solution)))
          #print("----!!!Solution found: %s!!!-----" % str(iteration_step))
          num_solutions+=1
      

    
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

    if solutions:
      print("\nSolution found on actual search depth, STOP")
      for solution in solutions:
        print("Solution: %s" % solution )
      break
    
    #last action is to set iterator conditions for next run
    itv_deep        += 1
    iterator_start  = 0       #if we resumed from a file then iter_start is not zero.



  print('Signalling all workers to terminate')  
  #send the kill flag, on reception each worker will return
  for proc in process_list:
    job_data = [0] * 3
    job_data[0] = -1
    q.put(job_data)

  #wait for all workers to return
  [proc.join() for proc in process_list]
  print("All worker-jobs finished")

  
  exit(0)



if __name__=='__main__':
  main()






