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
columns = ['Top' + str(i+1) for i in range(4)]
df['joined'] = df[columns].apply(lambda x: ' '.join(x.astype(str)), axis=1)
df1 = df[['Label', 'joined']].copy()

binary_labels=[]
for row in range(0,len(df1.index)):
	binary_labels.append(df1['Label'][row])

sentences=[]
#len(df1.index)
for row in range(0,len(df1.index)):
  sentence=df1['joined'][row]
  sentencel=preprocess(sentence)

	#print(preprocess(sentence))

  for i in sentencel:
    if i=='govts':
      i='government'
    elif i=='govt':
      i='government'
    elif i=='georgian' or i=='georgias':
      i='georgia'
    elif i=='american' or i=='americas':
      i='america'

  sentences.append(sentencel)
	

s=0
for sentence in sentences:
	x='b'
	remove_all(sentence,x)
	s=max(s,len(sentence))
print(s)
'''
fin=open('vocab.txt',"a")
kl=0
vocab_dict={}
for sentence in sentences:
  for tokens in sentence:
    if tokens in vocab_dict:
      kl=kl+1
    else:
      fin.write(str(tokens)+'\n')
      vocab_dict[tokens]=1
'''


vocab = open('vocab.txt').read().split('\n')
vocab_to_idx = dict([ (vocab[i],i) for i in range(len(vocab))])
idx_to_vocab=dict([(i,vocab[i]) for i in range(len(vocab))])
def sentence_to_indices(sentence):
  
  res = []
  for token in sentence:
    if token in vocab_to_idx:
      res.append(vocab_to_idx[token])
  return res

sentences1=[]
for sentence in sentences:
	sentence=sentence_to_indices(sentence)
	sentences1.append(sentence)
#sentences = map( sentence_to_indices, sentences)

#print(sentences[0])

X_train=[]
y_train=[]
for i in range(0,1900):     #change
	X_train.append(sentences1[i])
	y_train.append(binary_labels[i])
#y_train=binary_labels[0:90]
y_train=numpy.expand_dims(y_train,axis=1)
#print(y_train)
X_test=[]
y_test=[]
#len(df1.index)
for i in range(1900,len(df1.index)):   #change
	X_test.append(sentences1[i])
	y_test.append(binary_labels[i])


#print(X_test)
y_test=numpy.expand_dims(y_test,axis=1)



embeddings_idx = {}
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_idx[word] = coefs
f.close()
#print( len(embeddings_idx))

EMBEDDING_DIM=50
embedding_matrix = np.zeros(( 5079865 , EMBEDDING_DIM))
#print(embeddings_idx.get('pad'))
for word, i in vocab_to_idx.items():
    embedding_vector = embeddings_idx.get(word) #change
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector


for i in range(len(X_train)):
  kl=len(X_train[i])
  for j in range((s-kl)):
    X_train[i].append(0)

for i in range(len(X_test)):
  kl=len(X_test[i])
  for j in range((s-kl)):
    X_test[i].append(0)


model = Sequential()

embedding_layer = Embedding(5079865,      #change
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=s,
                            trainable=False)
model.add(embedding_layer)
model.add(Bidirectional(LSTM(8, dropout=0.2, recurrent_dropout=0.2)))  #change
dense1=Dense(1, activation='sigmoid')
model.add(dense1)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(np.array(X_train), np.array(y_train), epochs=7, batch_size=16)   #change      #########55.71% accuracy epochs 7 batch size 16

scores = model.evaluate(np.array(X_test), np.array(y_test), verbose=0)
print(scores)
print("Accuracy: %.2f%%" % (scores[1]*100))




'''
weights = dense1.get_weights()
print(weights)
for i in range(len(X_test)):
  for j in range(len(X_test[i])):
    if(X_test[i][j]!=0):
      print(idx_to_vocab[X_test[i][j]])
predictions=model.predict( X_test, batch_size=1, verbose=0, steps=None)
#print(predictions)
'''

















