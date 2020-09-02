from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf
import json
import pandas as pd
from csv import reader
from keras.preprocessing import text, sequence
from csv import DictWriter
from scipy.stats.mstats import gmean
import scipy as scp
from preprocessing import preprocess_func
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

def append_dict_as_row(file_name, dict_of_elem, field_names):
    with open(file_name, 'a+', newline='') as write_obj:     # Open file in append mode
        dict_writer = DictWriter(write_obj, fieldnames=field_names) # Create a writer object from csv module
        dict_writer.writerow(dict_of_elem) # Add dictionary as wor in the csv

TEXT_COLUMN = 'comment_text'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

input_str = 'bloody bastard fuck'
list_str = []
string = {'comment_text': input_str}
print("Original Text Input:", string['comment_text'])
string['comment_text'] = preprocess_func(string['comment_text'])
print("Preprocessed Text Input:", string['comment_text'])
list_str.append(string['comment_text'])
test_df = pd.DataFrame(columns=['comment_text'])
test_df['comment_text'] = list_str
#print(test_df['comment_text'])
x_test = test_df[TEXT_COLUMN].astype(str)
print("x_test:",x_test.loc[0])
tokenizer = text.Tokenizer(filters=CHARS_TO_REMOVE)
tokenizer.fit_on_texts(list(x_test))
x_test = tokenizer.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=220)

with open('model2.json', 'r') as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model1.h5")
    # predict
    pred_json = loaded_model.predict(x_test)
    print("Prediction:", pred_json)
    print("std", np.std(pred_json[1][0]))