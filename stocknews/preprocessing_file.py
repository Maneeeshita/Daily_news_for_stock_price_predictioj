import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
import string
#import matplotlib inline
'''
for row in range(0,len(data.index)):
    binary_labels.append(data.iloc[row,1])
'''


def preprocess(sentence):
	sentence = sentence.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	sentence1 = ''.join([i for i in sentence if not i.isdigit()])
	tokens = tokenizer.tokenize(sentence1)

	#filtered_words = filter(lambda token: token not in stopwords.words('english'), tokens)
	filtered_words = [w for w in tokens if not w in stopwords.words('english') ]
	return filtered_words

def remove_all(seq, value):
    pos = 0
    for item in seq:
        if item != value:
           seq[pos] = item
           pos += 1
    del seq[pos:]

df = pd.read_csv('Combined_News_DJIA.csv', parse_dates=True, index_col=0)
#print(df.head())
columns = ['Top' + str(i+1) for i in range(8)]
df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
df1 = df[['Label', 'joined']].copy()
#print(df1.head())

#print(df1['joined'][0])
train = df1.ix['2008-08-08':'2014-12-31']
test = df1.ix['2015-01-02':'2016-07-01']
binary_labels=[]
for row in range(0,len(df1.index)):
    binary_labels.append(df1['Label'][row])
#print(binary_labels)
sentences=[]
#len(df1.index)
for row in range(0,len(df1.index)):
	sentence=df1['joined'][row]
	#print(preprocess(sentence))
	sentences.append(preprocess(sentence))


for sentence in sentences:
	x='b'
	remove_all(sentence,x)

print(sentences)





