import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import tensorflow as tf
from nltk.corpus import stopwords

from tensorflow.keras import regularizers, initializers, optimizers, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Model

MAX_NB_WORDS = 100000    # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 200 # max length of each entry (sentence), including padding
VALIDATION_SPLIT = 0.20   # 20% data for validation (not used in training)
EMBEDDING_DIM = 300      # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "glove_files/glove.6B."+str(EMBEDDING_DIM)+"d.txt"

train = pd.read_csv('preprocessed_train.csv', encoding="utf8") # Reading Train data
train = train[:500000]

labels = ['severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'] # Defining Labels
y = train[labels].values # Storing values of labels in y
comments_train = train['comments_text'].astype(str)
comments_train = list(comments_train)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(comments_train)
sequences = tokenizer.texts_to_sequences(comments_train)

data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', y.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = y[indices]

num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])
x_train = data[: -num_validation_samples]
y_train = labels[: -num_validation_samples]
x_val = data[-num_validation_samples: ]
y_val = labels[-num_validation_samples: ]

embeddings_index = {}
f = open(GLOVE_DIR, encoding="utf8")
print('Loading GloVe from:', GLOVE_DIR,'...', end='')
for line in f:
    values = line.split()
    word = values[0]
    embeddings_index[word] = np.asarray(values[1:], dtype='float32')
f.close()

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print("Matrix Generated!")

print("Building Model!")
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights = [embedding_matrix],
                            input_length = MAX_SEQUENCE_LENGTH,
                            trainable=False,
                            name = 'embeddings')
embedded_sequences = embedding_layer(sequence_input)
x = LSTM(60, return_sequences=True,name='lstm_layer')(embedded_sequences)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
preds = Dense(5, activation="sigmoid")(x)

model = Model(sequence_input, preds)
model.compile(loss = 'binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])

tf.keras.utils.plot_model(model)
print('Training progress:')
history = model.fit(data, labels, epochs = 2, validation_data = (x_val, y_val), batch_size = 32, verbose = 1, use_multiprocessing = True)
scores = model.evaluate(data, labels, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# serialize model to JSON
#model_json = model.to_json()
#with open("struct_10L.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("weights_10L.h5")
#print("Saved model to disk")

test = "You're a bitch"
print("input:",test)
tokenizer = text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(list(test))
test = tokenizer.texts_to_sequences(test)
test = sequence.pad_sequences(test, maxlen=200)
prediction = model.predict(test)
print("Prediction:",prediction*100)

test = "I'll fuck you"
print("input:",test)
tokenizer = text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(list(test))
test = tokenizer.texts_to_sequences(test)
test = sequence.pad_sequences(test, maxlen=200)
prediction = model.predict(test)
print("Prediction:",prediction*100)

test = "I spit on his face"
print("input:",test)
tokenizer = text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(list(test))
test = tokenizer.texts_to_sequences(test)
test = sequence.pad_sequences(test, maxlen=200)
prediction = model.predict(test)
print("Prediction:",prediction*100)

loss = history.history['loss']
val_loss = history.history['val_loss']

lowest1 = min(loss)
lowest2 = min(val_loss)
f_lowest = min(lowest1,lowest2)

epochs = range(1, len(loss)+1)
plt.figure(figsize=(16,9))
plt.plot(epochs, loss, label='Training loss')
plt.plot(epochs, val_loss, label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.yticks(np.arange(f_lowest, max(loss), (max(loss) - min(loss))/30))
plt.xticks(np.arange(1,2.1,0.1))
plt.legend()
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
plt.figure(figsize=(16,9))
plt.plot(epochs, accuracy, label='Training accuracy')
plt.plot(epochs, val_accuracy, label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.xticks(np.arange(1,2.1,0.1))
#plt.yticks(np.arange(0.49, 0.99, (0.99 - 0.49)/20))
plt.legend()
plt.show()

