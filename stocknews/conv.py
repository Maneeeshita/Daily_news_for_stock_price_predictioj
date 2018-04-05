import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import nltk
import keras
import numpy 
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from keras.layers import LSTM, Convolution1D, Flatten, Dropout, Dense
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence



from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import re
import math
import string
#import matplotlib inline


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

#df = pd.read_csv('Combined_News_DJIA.csv', parse_dates=True, index_col=0)
df = pd.read_csv('Combined_News_DJIA.csv')
#print(df.head())
columns = ['Top' + str(i+1) for i in range(10)]
df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
df1 = df[['Label', 'joined']].copy()

binary_labels=[]
for row in range(0,len(df1.index)):
	binary_labels.append(df1['Label'][row])

sentences=[]
#len(df1.index)
for row in range(0,len(df1.index)):        #change
	sentence=df1['joined'][row]
	#print(preprocess(sentence))
	sentences.append(preprocess(sentence))

s=0
for sentence in sentences:
	x='b'
	remove_all(sentence,x)
	s=max(s,len(sentence))
#print('max_decode_length')
#print(s)

fin=open('vocab.txt',"a")

for sentence in sentences:
	for tokens in sentence:
		  fin.write(str(tokens)+'\n')

vocab = open('vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])

def sentence_to_indices(sentence):
  
  res = [0]
  for token in sentence:
    if token in vocab_to_idx:
      res.append(vocab_to_idx[token])
    else:
      res.append(2)
  res.append(1)
  return res

sentences1=[]
for sentence in sentences:
	sentence=sentence_to_indices(sentence)
	sentences1.append(sentence)
#sentences = map( sentence_to_indices, sentences)

#print(sentences[0])

X_train=[]
y_train=[]
for i in range(0,1700):     #change
	X_train.append(sentences1[i])
	y_train.append(binary_labels[i])
#y_train=binary_labels[0:90]
y_train=numpy.expand_dims(y_train,axis=1)
#print(y_train)
X_test=[]
y_test=[]
for i in range(1700,len(df1.index)):   #change
	X_test.append(sentences1[i])
	y_test.append(binary_labels[i])

y_test=numpy.expand_dims(y_test,axis=1)
embeddings_idx = {}

f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_idx[word] = coefs
f.close()
print( len(embeddings_idx))

EMBEDDING_DIM=50
embedding_matrix = np.zeros(( 6079865 , EMBEDDING_DIM))
for word, i in vocab_to_idx.items():
    embedding_vector = embeddings_idx.get(word) #change
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector


X_train = sequence.pad_sequences(X_train, maxlen=s)
X_test = sequence.pad_sequences(X_test, maxlen=s)


model = Sequential()

embedding_layer = Embedding( 6079865,      #change
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=s,
                            trainable=False)
model.add(embedding_layer)


model.add(Convolution1D(8, 3, border_mode='same'))
model.add(Dropout(0.2))

model.add(Convolution1D(4, 3, border_mode='same'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])




#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=7, batch_size=10)   #change      #########55.71% accuracy epochs 7 batch size 16

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


















