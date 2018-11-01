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
from keras.preprocessing.sequence import pad_sequences


from keras.models import model_from_json
import pickle

# Model reconstruction from JSON file
with open(r'saved_parameters\model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

# Load weights into the new model
model.load_weights(r'saved_parameters\model_weights.h5')

tokenizer = pickle.load(open(r'saved_parameters\tokenizer.pickle', 'rb'))

def generate_text(seed_text, next_words, max_sequence_len=20, model=model, tokenizer=tokenizer):
    predicted_indices = []
    next_words = int(next_words)
    for j in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
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

print(generate_text(sys.argv[1], sys.argv[2]))
