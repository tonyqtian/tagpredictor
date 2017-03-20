'''
Created on Mar 17, 2017

@author: tonyq
'''
import logging
import sys, os, errno
import re

#-----------------------------------------------------------------------------------------------------------#

def setLogger(timestr, onscreen=True, out_dir=None):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)
	if onscreen:
		console_format = BColors.OKBLUE + '[%(levelname)s]' + BColors.ENDC + ' (%(name)s) %(message)s'
		#datefmt='%Y-%m-%d %Hh-%Mm-%Ss'
		console = logging.StreamHandler(sys.stdout)
		console.setLevel(logging.DEBUG)
		console.setFormatter(logging.Formatter(console_format))
		logger.addHandler(console)
	if out_dir:
		file_format = '[%(levelname)s] (%(name)s) %(message)s'
		log_file = logging.FileHandler(out_dir + '/' + timestr + 'log.txt', mode='w', encoding='UTF8')
		log_file.setLevel(logging.DEBUG)
		log_file.setFormatter(logging.Formatter(file_format))
		logger.addHandler(log_file)

#-----------------------------------------------------------------------------------------------------------#

def mkdir(path):
	if path == '':
		return
	try:
		os.makedirs(path)
	except OSError as exc: # Python >2.5
		if exc.errno == errno.EEXIST and os.path.isdir(path):
			pass
		else: raise
		
class BColors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'
	WHITE = '\033[37m'
	YELLOW = '\033[33m'
	GREEN = '\033[32m'
	BLUE = '\033[34m'
	CYAN = '\033[36m'
	RED = '\033[31m'
	MAGENTA = '\033[35m'
	BLACK = '\033[30m'
	BHEADER = BOLD + '\033[95m'
	BOKBLUE = BOLD + '\033[94m'
	BOKGREEN = BOLD + '\033[92m'
	BWARNING = BOLD + '\033[93m'
	BFAIL = BOLD + '\033[91m'
	BUNDERLINE = BOLD + '\033[4m'
	BWHITE = BOLD + '\033[37m'
	BYELLOW = BOLD + '\033[33m'
	BGREEN = BOLD + '\033[32m'
	BBLUE = BOLD + '\033[34m'
	BCYAN = BOLD + '\033[36m'
	BRED = BOLD + '\033[31m'
	BMAGENTA = BOLD + '\033[35m'
	BBLACK = BOLD + '\033[30m'
	
	@staticmethod
	def cleared(s):
		return re.sub("\033\[[0-9][0-9]?m", "", s)

def red(message):
	return BColors.RED + str(message) + BColors.ENDC

def b_red(message):
	return BColors.BRED + str(message) + BColors.ENDC

def blue(message):
	return BColors.BLUE + str(message) + BColors.ENDC

def b_yellow(message):
	return BColors.BYELLOW + str(message) + BColors.ENDC

def green(message):
	return BColors.GREEN + str(message) + BColors.ENDC

def b_green(message):
	return BColors.BGREEN + str(message) + BColors.ENDC