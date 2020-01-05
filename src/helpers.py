import os

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



def main():
  return

if __name__=='__main__':
  main()