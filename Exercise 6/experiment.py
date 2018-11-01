import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import codecs

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Dropout, Embedding
from keras.layers.recurrent import LSTM, GRU, Recurrent
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
import re

def load_file(file_path):
    # Load the file, returning as a string object
    # For example maybe something like: text = codecs.open(file_path, encoding='utf-8').read().lower() ?
    text = codecs.open(file_path, encoding='utf-8').read().lower()
    # Return text, unique chararacters, and words
    return text

def text_pre_precessing(text):
    # Clean the text data
    # For example, you may want to remove all the punctuatiation and split by words
    # or any other pre-processing.

    text = str(text)
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    #text = text.split()

    return text

def prepare_dataset(text, length, stride, char_to_int):
    # Here generate sentences from the word-splited text
    # For example given a text "...the nameless city when i drew nigh the nameless city i knew..." with length=4, stride=2
    # it should return "the nameless city when", "city when i drew", "i drew nigh the", "nigh the nameless city", ...
    #create mapping of unique chars to integers
    dataX = []
    dataY = []
    for i in range(0, len(text)-length, stride):
        seq_in = text[i:i + length]
        seq_out = text[i+length]
        dataX.append([char_to_int[char] for char in seq_in])
        dataY.append(char_to_int[seq_out])
    print('Total patterns:', len(dataX))
    return dataX, dataY, len(dataX)



text = load_file('lovecraft.txt')

text = text_pre_precessing(text)  

chars = sorted(list(set(text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)
print("total characters:", n_chars)
print("total vocab:", n_vocab) #number of distinct characters

dataX, dataY, n_patterns = prepare_dataset(text, 100, 1, char_to_int)
#reshape X to be [samples, time_steps, features]
X = np.reshape(dataX, (n_patterns, 100, 1))
#rescale the integers to the range 0-to-1
X = X / float(n_vocab)
#one hot encode the output variable
y = np_utils.to_categorical(dataY)

#rand_ind = np.random.randint(0, X.shape[0]-1, 120000)
num_train = 120000
train_x = X[0:num_train, :, :]
train_y = y[0:num_train,:]
test_x = X[num_train:X.shapep[0],:,:]
test_y = y[num_train:y.shape[0],:]

#define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]))) #(100,1)
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

model.fit(train_x, train_y, epochs=20, validation_data=(test_x, test_y), batch_size=128)
