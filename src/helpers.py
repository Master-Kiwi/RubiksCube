#CHANGELOG: helpers.py
#AUTHOR: SL

#12.01.2020 
#  initial version
#  implementes helper functions with main purpose to get smarter output on console
#  TODO: 
#    check for terminal cursormove actions to redraw more then 1 line with \r
#    would be nice to redraw a colored cube to watch the solution progress


import os
import time
import sys, platform

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

#experimental, TODO: check if works on pycharm
def console_supports_ansi():
  if (hasattr(sys.stdout, "isatty") and sys.stdout.isatty()):
    return True
  else:
    print("ANSI output disabled")
    return False

#draws the ansi-color table, if bg=fg it is invisible
#use this if your color output looks weird
def console_print_color_table():
  #each line starts with ESC+CSI following the SGR Code; ESC="\x1b" is dez 27 or octal 33
  CSI = "\x1b["
  
  #with this ASCII sequence the style is reset to default
  end_str=CSI+"0m"
  
  #SGR Code ist style; fg; bg
  #if we do not reset the format with end_str then multiple styles like bold and underline are stacked up
  #not all styles are supported
  #most basic should be style 0 or 1
  for style in range(0,8):
    for fg in range(30, 38):
      for bg in range(40, 48):
        str="%s%d;%d;%dm" % (CSI,style,fg, bg)
        print("%s %s %s"  %(str, str[2:], end_str) , end = "")
      print("")
    print("")

#STYLES:
#0	Reset / Normal	all attributes off
#1	Bold or increased intensity	
#2	Faint (decreased intensity)	
#3	Italic	Not widely supported. Sometimes treated as inverse.
#4	Underline	
#5	Slow Blink	less than 150 per minute
#6	Rapid Blink	MS-DOS ANSI.SYS; 150+ per minute; not widely supported
#7	Reverse video	swap foreground and background colors
  
#replace this strings in RubiksCube.py 
  #col_str_black   = '\x1b[1;37;40m' 
  #col_str_red     = '\x1b[1;37;41m'
  #col_str_green   = '\x1b[1;37;42m'
  #col_str_yellow  = '\x1b[1;37;43m'
  #col_str_blue    = '\x1b[1;37;44m'
  #col_str_orange  = '\x1b[1;37;45m'
  #col_str_white   = '\x1b[1;37;47m'
  #col_str_end     = '\x1b[0m'
  

def main():
  if console_supports_ansi():
    print("Console supporting ANSI")
  else:
    print("ANSI output disabled")

  os.system('color') 
  console_print_color_table()
  print("Press Any Key to continue")
  input()

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