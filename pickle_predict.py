import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from preprocessing import preprocess_func
import pandas as pd
from keras.preprocessing import text, sequence
import pickle

TEXT_COLUMN = 'comment_text'
CHARS_TO_REMOVE = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n“”’\'∞θ÷α•à−β∅³π‘₹´°£€\×™√²—'

pickle_model_name = 'pi.sav'
input_str = 'happy people'
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
print("Loading Pickle Model...")
pickle_loaded_model = pickle.load(open(pickle_model_name, 'rb'))
result = pickle_loaded_model.predict(x_test)
print("Pickle Prediction:", result)