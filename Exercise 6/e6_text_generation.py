import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import os
import codecs

import keras
from keras.layers import Input, Dense, Activation, Dropout, Embedding
from keras.layers.recurrent import LSTM, GRU, Recurrent
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import to_categorical
import re
from keras.preprocessing.sequence import pad_sequences

from google.colab import files

uploaded = files.upload()


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

    text = text.split()

    return text

def generate_sentences(text, length, stride):
    # Here generate sentences from the word-splited text
    # For example given a text "...the nameless city when i drew nigh the nameless city i knew..." with length=4, stride=2
    # it should return "the nameless city when", "city when i drew", "i drew nigh the", "nigh the nameless city", ...
    #create mapping of unique chars to integers
    sentences = []
    for i in range(0, len(text)-length, stride):
        sentences.append(text[i:i+length])
    return sentences

def vectorize_sentences(sentences, t):
    # This will vectorize or binarize your sentences.
    # One lazy method is to use Tokenizer API of Keras, but actually recommended.
    t.fit_on_texts(sentences)
    total_words = len(t.word_index) + 1   #total_words = np.amax(sentences)+1; the number of unique words
    sentences = t.texts_to_sequences(sentences) 

    return np.array(sentences), total_words

def data_prepare(sentences):
    # We have some unprepared data with shape (num_sentences, length),
    # we want to split it into train_x: (num_sentences, length - 1), train_y: (num_sentences, 1)
    # also some test_x and test_y
    num_sentences = sentences.shape[0]
    num_train = int(np.round(num_sentences * 0.8))
    num_test = num_sentences - num_train
    Y = sentences[:, -1]
    Y = np_utils.to_categorical(Y, num_classes = total_words)

    train_x = sentences[:num_train, :-1]
    train_y = Y[0:num_train]
    
    test_x = sentences[num_train:num_train+num_test, :-1]
    test_y = Y[num_train:num_sentences]

    return train_x, train_y, test_x, test_y     # don't forget to to_categorical the labels


text = load_file('wonderland.txt')

chars = sorted(list(set(text)))
char_to_int = dict((c,i) for i, c in enumerate(chars))

n_chars = len(text)
n_vocab = len(chars)
print("total characters:", n_chars)
print("total vocab:", n_vocab) #number of distinct characters

t = Tokenizer()

text = text_pre_precessing(text)    
sentences = generate_sentences(text, 20, 2)
sentences, total_words = vectorize_sentences(sentences, t)
train_x, train_y, test_x, test_y = data_prepare(sentences)

input = Input(shape=(train_x.shape[1],))

#This is the size of the vocabulary in the text data.
#For example, if your data is integer encoded to values between 0-10,
#then the size of the vocabulary would be 11 words.
#here the input size I define it as np.amax(sentences)+1 = total_words
x = Embedding(total_words, 100, input_length = sentences.shape[1]-1)(input) #input_length should coincide with train_x.shape[1], which is 19 instead of 20
x = LSTM(256, input_shape=(train_x.shape[1], train_x.shape[0]))(x)
x = Dense(512)(x)
x = Dense(2678, activation='softmax')(x)

model = keras.Model(input, x)

model.compile(loss='categorical_crossentropy', optimizer='adam')

model.summary()

model.fit(train_x, train_y, validation_data = (test_x, test_y), epochs = 10, batch_size = 128)

model.save('saved_model.h5')  # Most important thing, always save your model !!!!!!!

def generate_text(seed_text, next_words, max_sequence_len, model, tokenizer):
    predicted_indices = []
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        print(token_list)
        token_list = pad_sequences([token_list], maxlen= max_sequence_len-1, padding='pre')
        predicted = model.predict(token_list)
        predicted_index = np.argmax(predicted[0,:])
        predicted_indices.append(predicted_index)
  
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text
	
predicted = generate_text("Alice laughed so much at this", 5, 20, model, t)
print(predicted)