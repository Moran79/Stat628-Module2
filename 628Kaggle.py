# Import Essential Packages
import json
import sklearn
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
import textblob
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
import nltk.sentiment
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pyitlib import discrete_random_variable as drv
import time
import os

# Feature List
embeddings_index = {}
f = open(os.path.join('/Users/moran/Google_Drive/Course/628/Proj2/data/glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# Prepare train set:

def change_to_vector(text):
    tmp_list = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)][-360:]
    outfile = np.zeros((360), dtype='int32')
    for idx, word in enumerate(tmp_list):
        outfile[idx] = wordsList.index(word) if word in wordsList_set else 399999
    return outfile

#1000 data / 25s
start = time.time()
ids = np.zeros((1000, 360), dtype='int32')
y = []
with open('/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json', 'r') as fh:
    for idx,line in enumerate(fh):
        d = json.loads(line)
        text = d['text']
        y.append(d['stars'])
        ids[idx,:] = change_to_vector(text).copy()
end = time.time()
print(end - start)


### LSTM


#package for RNN model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GRU
from keras.layers import Input
from keras.models import Model


pd.options.mode.chained_assignment = None



def get_dummy(array):
    res = np.zeros((len(array),5))
    for i in range(len(array)):
        res[i, int(array[i]) - 1] = 1
    return res

def get_star(array):
    nr = array.shape[0]
    res = []
    for i in range(nr):
        res.append(np.where(array[i] == 1)[0][0] + 1)
    return res
# create the model






from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 360
tokenizer = Tokenizer(num_words = MAX_NB_WORDS)
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = df['stars']
# labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# split the data into a training set and a validation set
VALIDATION_SPLIT = 0.2
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]

x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]


start = time.time()
EMBEDDING_DIM = 50
end = time.time()
print(end - start)
EMBEDDING_DIM = 50
embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector



### Readin Data

ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)
text = df['text']



#Model-1
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ids,y_dummy ,test_size=0.2)

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)


inputs = Input(shape=(x_train.shape[1], ), dtype='int32')
embedded_sequences = embedding_layer(inputs)
x = LSTM(16,dropout=0.2, recurrent_dropout=0.2,implementation=1)(embedded_sequences)
output = Dense(5, activation="sigmoid")(x)
model = Model(inputs, output)
model.compile(loss='mse',optimizer='adam',metrics=['acc'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=2, batch_size=128)

# record the result
prediction = model.predict(X_test,verbose=0)

reuslt1 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
index1 = reuslt1>5
reuslt1[index1] = 5
rmse1 = np.sqrt(np.sum(np.square(reuslt1 - y))/5000)

prediction = model.predict(X_train,verbose=0)

reuslt2 = prediction[:,0] + 2*prediction[:,1] + 3*prediction[:,2] + 4*prediction[:,3] + 5*prediction[:,4]
index2 = reuslt2>5
reuslt2[index2] = 5
rmse2 = np.sqrt(np.sum(np.square(reuslt1-get_star(y_test)))/200)



