import numpy as np
import math
import string

fin=open('INPUT.txt').read().split('\n')
for line in fin:
	if(line):
		tokens=line.split(' ')
		tokens[0]=''
fin.close()
