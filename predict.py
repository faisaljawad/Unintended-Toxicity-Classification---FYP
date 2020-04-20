# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 17:18:25 2020

@author: faisa
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
tqdm.pandas()
from keras.preprocessing import text, sequence

model = tf.keras.models.load_model('model.h5')

x_test = "you are cute"
print("input:",x_test)
tokenizer = text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(list(x_test))
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=200)
prediction = model.predict(x_test)
print(prediction*100)