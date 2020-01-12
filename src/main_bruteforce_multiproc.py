#CHANGELOG: main_multiproc.py
#AUTHOR: SL

#12.01.2020 
#  changed name to: "main_bruteforce_multiproc.py"
#  improved visualization using helper functions requires helpers.py from 12.01.2020
#  major changes to job-dispatching
#   there was a bug with dispatching implemented before, as all jobs have been queued at the beginning.
#   with high number of jobs like on depth > 12 it caused some GB of RAM allocated
#  it was changed in a way that only up to  target_jobs = num_process * 100  can be queued
#  when this queue is full or no further jobs are available then we wait for responses from the worker pool
#  each response decrements jobs_to_complete number --> jobs_to_complete = 0 is indication for all batches have been computed
#  if we do not get a response within a timeout it means that alle workers are still busy or all workers have completed.
#  so on timeout we start dispatching again
#  it could be seen that there is a variation in worker_jobs computation time, especicially at the start there are many iterations thtat are skipped
#    this lead to a case where the responses come that fast that timeout occurs first when actual_jobs = 0
#    therefore target_jobs is a relative high number to quarantee a good CPU utilization in the case when.
#    it is hard to find a solution fitting vor variation of CPU-speed and CPU-cores
#    on weird computation times change the batch_size (lower batch size will give you faster responses) or change the timeout to wait for responses
#   there is a benchmark function running at beginning that estimates your system performance, you can use that number as an indicator for batch_size

#this is source code for the section where responses are waited for
  #poll until all jobs done 
  #while jobs_to_complete > 0:
  #  data_available = parent_conn.poll(timeout=0.2) #wait for data with timeout
  #  #on timeout break, timeout occurs if queue gets empty or all worker jobs are computing a batch and have not completed within timeout
  #  if data_available == False:
  #    break
  #  else:
  #    ....

    


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
import random
from copy import copy, deepcopy
from multiprocessing import Pool, Process, Queue, Pipe, Lock

#self written submodules
from RubiksCube import tRubikCube
from iterate_tools import tIterate
from helpers import console_clear, num_to_str_si, sec_to_str



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
    skipped_iter_steps = 0
    #print("PID: %d  Job data: %s" %(os.getpid(), str(job_data)))
    #compute iteration batch
    for iter in range(batch_size):
      iteration_step = next(iter_func)           #using generator is faster
      
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
      Test = tRubikCube()
      #Perform all rotate actions, iteration_step is a list of actions, list comprehension is faster
      [Test.actions_simple(action) for action in iteration_step]
      #check for match with target, we need to do this only after the rotation is complete, as we checked all other variants before
      if Test.equals(Cube): 
        #print("----!!!Solution found: %s!!!-----" % str(iteration_step))
        #Solved_Cubes.append(deepcopy(Test))
        
        #notify manager process, send result, last iteration number and sequence if solution is found
        conn.send([True, iter_start+iter, iteration_step, skipped_iter_steps])
 
    conn.send([False, iter_start+batch_size, [], skipped_iter_steps])

  #print('Exiting worker-function')
  #conn.close()

    
def main():
  #enable colored text on console outputs
  os.system('color') 
  #increase size of console
  os.system('mode con: cols=160 lines=40')  #12*4 +1

  Cube = tRubikCube()
  
  #------load a prerotated cube-----------
  #filename = "../data/simple_cube_04.json"
  filename = "../data/moderate_cube_08.json"
  #filename = "../data/complex_cube_12.json"
  #filename = "../data/real_cube.json"

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
  if not Cube.actions_list: 
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
  #exists = False
  if exists:
    print("Found a Progress File: '%s'- Progress Statistics: " % progress_filename)
    resume_iterator = tIterate(filename = progress_file_path)  
    iter_steps    = resume_iterator.get_total_num()
    iter_step     = resume_iterator.get_step()
    print("Depth:  %03d" %resume_iterator.depth, end= "")
    print("  Num_Actions:  %03d" %resume_iterator.num_actions, end="")
    print("  Iteration:  %s/%s" %(num_to_str_si(iter_step), num_to_str_si(iter_steps)), end="")
    print("  Progress:  %.3f%%" % float(iter_step / iter_steps * 100.0))
    
    itv_deep        = resume_iterator.depth
    num_actions     = resume_iterator.num_actions 
    iterator_start  = iter_step

  else:
    print("No Progress File, start from beginning")
    itv_deep        = 1
    num_actions     = 12
    iterator_start  = 0

  #cpu-count -1 workers, 1 cpu slot left for manager
  num_process = os.cpu_count()-1
  #num_process = os.cpu_count()-8
  #num_process = 1
  
  #determin initial iter_per_sec value
  iter_per_sec = 0
  test_batch_size = 20000
  while iter_per_sec == 0:
    #determin initial iter per sec by short benchmark
    iter_per_sec = benchmark(itv_deep, test_batch_size)
    #if no result increase batch size
    test_batch_size *= 2
  
  #estimated calculation power, assume that each process can do the workload
  iter_per_sec *= num_process
  print("Estimated iter/sec (using %d CPUs): %s" %(num_process, num_to_str_si(iter_per_sec)))

  print("Press Enter to Start")
  input()
  console_clear()


  print("Search Solution with Iterative Deepening")
  print("Starting Multiple Worker Processes...")
  print("Using %d CPUs"%num_process) 
  q = Queue()
  l = Lock()
  parent_conn, child_conn = Pipe()

  #info('main-job')
  process_list = [Process(target=worker_process, args=(cube_file_path, q, l, child_conn)) for i in range(num_process)]
  [proc.start() for proc in process_list]
  time.sleep(2)
  
  solutions = []



  #search solution, increase depth after each run
  while True:
    start = datetime.datetime.now()     #iteration depth starting time
    last_save_time = start              #set last save timeestamp to now
    time.sleep(0.2)                     #guarantee that first time-delta is > 0 (for small iteration batches)
    
    #this iterator is used for saving progress
    save_iterator   = tIterate(itv_deep, num_actions, iterator_start)
    

    #Iterator and print Info
    seq_iterator      = tIterate(itv_deep, num_actions, iterator_start)
    #total number of iterations when starting from 0
    total_iter_steps  = seq_iterator.get_total_num()
    #Iteration steps to compute depend on iterator starting value
    iter_steps        = total_iter_steps - iterator_start 
    #estimated time to coplete
    time_to_completion = float(iter_steps / iter_per_sec)


    #output info for actual Iteration Depth
    print("\nDepth:  %d" % itv_deep, end='')
    print("  Timestamp: %s" % str(start), end="")
    print("  Iteration: %s/%s" % (num_to_str_si(iterator_start), num_to_str_si(iter_steps)), end='')
    print("  Estimated Time: %s" % sec_to_str(time_to_completion) )

    
    #define initial batch size
    batch_size = 20000
    #very low number of iterations - only one batch
    if(iter_steps < batch_size):  batch_size = iter_steps    

    #rounding up integer division typical yields a float value
    #(numerator + demoninator - 1) / demoninator)
    jobs_to_complete = int((iter_steps - iterator_start + batch_size - 1) / batch_size) 
    #this is number of jobs that need to complete
    #it is decremented when an result from the worker pool is received

    actual_queued_jobs = 0
    target_jobs = num_process * 100
    iter_start = iterator_start

    num_solutions=0
    last_iter = 0
    skipped_iter_steps = 0

    #as long as there are jobs pending
    while jobs_to_complete > 0:
      #dispatch jobs to saturate workers, do not dispatch too much as this will waste memory
      #dispatch a multiple of processes
      #print("Dispatching Jobs start...")
      #here we cannot use jobs_to_complete as break condition, as this will be used on reception to check if we received all answers
      #so we use the iterator to check 
      while iter_start < total_iter_steps and actual_queued_jobs < target_jobs:
        #build worker job starting data
        job_data = [0] * 3
        job_data[0] = int(iter_start)
        job_data[1] = int(itv_deep)
        #truncate last batch, it would be no problem to calculate more as iterator is restarting from beginning
        #but it might yield double results on high batch_size
        if iter_start+batch_size > total_iter_steps:
          job_data[2] = int(total_iter_steps - iter_start)
        else: 
          job_data[2] = int(batch_size)
        #send to queue, any of the workers will receive and process
        q.put(job_data, block=True, timeout=None)   #send blocking, it blocked never in many tries
        #typical all jobs are queued immediately
        actual_queued_jobs += 1

        iter_start += batch_size
        #print("\rQueued Jobs: %03d" % actual_queued_jobs, end="")


      #poll until all jobs done 
      while jobs_to_complete > 0:
        data_available = parent_conn.poll(timeout=0.2) #wait for data with timeout
        #print("\nTotal Jobs to complete: %d,    Actual Jobs in queue: %d" % (jobs_to_complete, actual_queued_jobs))
        #on timeout break, timeout occurs if queue gets empty or all worker jobs are computing a batch nad do not complete within timeout
        if data_available == False:
          break
        else:
          batch_result = parent_conn.recv()   #blocking, wait for recepetion, will not block here as we checked before with poll()
          actual_queued_jobs -= 1                                #decrement queued_jobs
          jobs_to_complete -= 1

          result =              batch_result[0]   #True if solution is found
          iter =                batch_result[1]   #last iteration number or actual iteration number if result=True
          iteration_step =      batch_result[2]   #empt list [] or action sequence if result=True
          skipped_iter_steps += batch_result[3]   #number of skipped iterations for this batch 

          actual = datetime.datetime.now()

          delta_save_time = actual - last_save_time
          delta_save_time = delta_save_time.total_seconds() 
          #check if progress needs to be saved
          #we save not the actual iteration-number, we save the old iteration number.
          #this is done to prevent from missing iterations, because multiple worker jobs deliver their results unordered
          #the jobs are ordered qued, but the answers will come unordered
          if delta_save_time > float(PROGRESS_SAVE_ITV):
            #print("\nSave Progress: %s" %progress_filename, end="")
            #print("  Iteration Number: %d" % save_iterator.get_step(), end="")
            #print("  Timestamp: %s" % str(last_save_time))
            save_iterator.save_to_file(progress_file_path)     #save to file
          
            #only the iteration step will increment, depth will increment on restart of outer-loop
            save_iterator.set_step(iter)
            last_save_time = actual     #set last save time
        
          #Print Progress Indication
          actual_delta = actual-start
          actual_delta = actual_delta.total_seconds() 
          if(actual_delta > 0.0) and (iter > last_iter):
            last_iter = iter
            iter_per_sec = (iter-iterator_start) / actual_delta
            progress = 100.0*iter/iter_steps
            remaining = iter_steps-iter
            time_to_completion = remaining / iter_per_sec
            print("\r[Progress=%.3f%%  iter=%s/%s  iter/sec=%s  remain=%s  skip=%s  jobs:%s/%s  save in %s]         " % 
                  (progress, num_to_str_si(iter), num_to_str_si(iter_steps), num_to_str_si(iter_per_sec),  
                  sec_to_str(time_to_completion), num_to_str_si(skipped_iter_steps), 
                  num_to_str_si(actual_queued_jobs), num_to_str_si(jobs_to_complete), 
                  sec_to_str(PROGRESS_SAVE_ITV-delta_save_time) ), end='', flush=True)
        
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
    print(" Total time = %s" % sec_to_str(delta_seconds))
    
    #iter_per_sec estimation should be correct as this is needed for runtime estimation upon start
    if delta_seconds > 0.0: 
      iter_per_sec = iter_steps / delta_seconds
      print(" Iterations per seconds = %s" % num_to_str_si(int(iter_per_sec)))
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
    
    #on completion of this run we reset the progress file
    save_iterator   = tIterate(itv_deep, num_actions, iterator_start)
    save_iterator.save_to_file(progress_file_path)     #save on each startup - means on each co


  print('Signalling all workers to terminate')  
  #send the kill flag, on reception each worker will return
  for proc in process_list:
    job_data = [0] * 3
    job_data[0] = -1
    q.put(job_data)

  #wait for all workers to return
  [proc.join() for proc in process_list]
  print("All worker-jobs finished")


#simple benchmarking with cube bruteforce algorithm
#iter per sec varies with itv_deep
def benchmark(itv_deep, batch_size):
  #Total Num of iterations
  iterator = tIterate(itv_deep, 12, 0)
  iter_steps   = iterator.get_total_num()
 
  random.seed()
  rand_start= int(random.randrange(iter_steps/4, iter_steps*3/4))   #get a random iterator number betwen 1/4 and 3/4 of possible
  #init iterator with random starting value
  seq_iterator = tIterate(itv_deep, 12, rand_start)
  iter_func    = seq_iterator.generator()

  Cube = tRubikCube()
  skipped_iter_steps = 0
  
  print(" <Single-Core-Benchmarking> Start at %s with %s iterations @ depth %d" %(num_to_str_si(rand_start), num_to_str_si(batch_size), itv_deep))
  
  start = datetime.datetime.now()     #iteration  starting time
  #compute iteration batch
  for iter in range(batch_size):
    iteration_step = next(iter_func)           #using generator is faster

    #ignore the skipped section, as they compute way faster and it will give unreliable result on high depths
    #  with approx 100M skipped iters at once

    ##check for unnecessary actions
    #last_action = iteration_step[0]
    #action_counter = 1
    #skip_step = False
    #for i in range(1, len(iteration_step)):
    #  action = iteration_step[i]
    #  if action == Cube.conj_action(last_action):
    #    skip_step = True
    #    break
        
    #  if action == last_action:
    #    action_counter += 1        
    #    if action_counter == 3:
    #      skip_step = True
    #      break
    #  else: 
    #    action_counter = 1
    #  last_action = action
      
    ##no need to compute this sequence
    #if skip_step:
    #  skipped_iter_steps += 1
    #  continue

    Test = tRubikCube()
    #Perform all rotate actions, iteration_step is a list of actions, list comprehension is faster
    [Test.actions_simple(action) for action in iteration_step]
    #check for match with target,
    Test.equals(Cube)
  
  stop = datetime.datetime.now()
  delta = stop-start
  delta_seconds = float(delta.total_seconds())
  print("  <Single-Core-Benchmarking> Skipped    = %s" % num_to_str_si(skipped_iter_steps))
  print("  <Single-Core-Benchmarking> Total time = %s" % sec_to_str(delta_seconds))
    
  #iter_per_sec estimation should be correct as this is needed for runtime estimation upon start
  if delta_seconds > 3.0: 
    iter_per_sec = batch_size / delta_seconds
    print("  <Single-Core-Benchmarking> Iter/sec   = %s" % num_to_str_si(int(iter_per_sec)))
    return int(iter_per_sec)
  else:
    print("  <Single-Core-Benchmarking> Iter/sec   = <not reliable, to short>")
    return 0


if __name__=='__main__':
  main()






