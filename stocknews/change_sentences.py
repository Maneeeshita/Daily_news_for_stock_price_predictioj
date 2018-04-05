import numpy as np
import math
import string
import pandas as pd 
import nltk

fin=open('final_input_data.txt').read().split('\n')
data=pd.read_csv('Combined_News_DJIA.csv')
all_labels=[]
c=0
for line in fin:
	if(line):
		#tokens = [word for sent in sent_tokenize(line) for word in word_tokenize(sent)]
		tokens=line.split(' ')
		s=[]

		c=c+1
		for i in tokens:
			if(i != ' ' or i!=',' or i!=''):
				s.append(i)
		all_labels.append(s[1:])
		#train_labels.append(tokens[1:])
print(c)
binary_labels = []
for row in range(0,len(data.index)):
    binary_labels.append(data.iloc[row,1])
print(binary_labels)

print(len(all_labels))

