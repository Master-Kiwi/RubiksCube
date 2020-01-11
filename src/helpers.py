import os
import time

# define our clear function for console 
def console_clear(): 
  # for windows 
  if os.name == 'nt': 
    _ = os.system('cls') 
  # for mac and linux(here, os.name is 'posix') 
  else: 
    _ = os.system('clear') 

# print process info
def info(title):
  print(title, end = "")
  print('\t  module name: %s' % __name__, end = "")
  if hasattr(os, 'getppid'):  # only available on Unix
    print('  parent process: %d' % os.getppid(), end="" )
  print('  process id:   %d' % os.getpid() )    

#converts a number into a str with SI prefix
#useful for displaying large numbers in small strings
#bounds are each 10*SI, that means 10000=10k is the, Number is maximum 4 digits except >= 10000*Y
def num_to_str_si(num):
  num = int(num)
  if num >= 10*1000*1000*1000*1000*1000*1000*1000*1000:  #septillion or Yotta 10^24
    str = "%dY" % (num/(1000*1000*1000*1000*1000*1000*1000*1000))
    return str
  if num >= 10*1000*1000*1000*1000*1000*1000*1000:       #sextillion or Zetta 10^21
    str = "%dZ" % (num/(1000*1000*1000*1000*1000*1000*1000))
    return str
  if num >= 10*1000*1000*1000*1000*1000*1000:            #quintillion or Exa  10^18
    str = "%dE" % (num/(1000*1000*1000*1000*1000*1000))
    return str
  if num >= 10*1000*1000*1000*1000*1000:                 #quadrillion or Penta
    str = "%dP" % (num/(1000*1000*1000*1000*1000))
    return str
  if num >= 10*1000*1000*1000*1000:                      #Trillion or Terra
    str = "%dT" % (num/(1000*1000*1000*1000))
    return str
  if num >= 10*1000*1000*1000:                           #Billion or Giga
    str = "%dG" % (num/(1000*1000*1000))
    return str
  if num >= 10*1000*1000:                                #Million or Mega
    str = "%dM" % (num/(1000*1000))
    return str
  if num >= 10*1000:                                     #Thousand or Kilo
    str = "%dk" % (num/1000)
    return str
  str = "%d" % num
  return str

#converts a number representing a second to a time-duration.
#from msec to Millenium
#useful for displaying large time durations in small strings
def sec_to_str(sec):
  sec = float(sec)
  if sec >= 1000*365*24*60*60: #1Millenium
    str = "%.2fmill" % (sec/(3600.0 * 24 * 365 * 1000))
    return str
  if sec >= 1*365*24*60*60: #1year
    str = "%.2fyrs" % (sec/(3600.0 * 24 * 365))
    return str
  if sec >= 2*24*60*60: #2days
    str = "%.2fdays" % (sec/(3600.0 * 24))
    return str
  if sec >= 2*60*60:   #2hours
    str = "%.2fhrs" % (sec/3600.0)
    return str
  if sec >= 2*60:     #2min
    str = "%.2fmin" % (sec/60.0)
    return str
  if sec <= 1.0:
    str = "%.0fms" % (sec*1000.0)
    return str
  str = "%.2fsec" % sec
  return str

def main():
  num = 1
  sec = 0.1
  while True:
    num *= 5
    sec = sec + sec * 0.2
    #print("\r Number: %s      " %num_to_str_si(num), end= "")
    print("\r Time: %s      " %sec_to_str(sec))
    time.sleep(0.1)
  return

if __name__=='__main__':
  main()