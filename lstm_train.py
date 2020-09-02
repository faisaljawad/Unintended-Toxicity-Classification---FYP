import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate
from keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.utils import shuffle
from keras.layers import Dropout
import smart_open

MAX_WORDS = 100000
NUM_MODELS = 1
BATCH_SIZE = 64
LSTM_UNITS = 60
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
EPOCHS = 5
MAX_LEN = 220
checkpoint_predictions = []
weights = []

EMBEDDING_FILES = ['glove_files/glove.840B.300d.gensim']

IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
                    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']

AUX_COLUMNS = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
TEXT_COLUMN = 'comments_text'
TARGET_COLUMN = 'target'

train_df = pd.read_csv('preprocessed_train.csv', encoding="utf8")
test_df = pd.read_csv('preprocessed_test.csv', encoding="utf8")
#train_df = train_df[:1000]
#test_df= test_df[:10]
train_df = shuffle(train_df) # shuffling

print("Dataset loaded...")

def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)
    
def convert_dataframe_to_bool(df):
    bool_df = df.copy()
    for col in [TARGET_COLUMN ] + IDENTITY_COLUMNS:
        convert_to_bool(bool_df, col)
    return bool_df

x_train = train_df[TEXT_COLUMN].astype(str) # storing comment text from train dataframe
y_train = train_df[TARGET_COLUMN].values # storing target column from train dataframe
y_aux_train = train_df[AUX_COLUMNS].values # storing auxilary column train values
x_test = test_df['comment_text'].astype(str) # storing comment text from test dataframe

train_df = convert_dataframe_to_bool(train_df) # converting all values into zeros and ones with threshold 0.5

tokenizer = Tokenizer(num_words = MAX_WORDS) # Initializing Tokenizer
tokenizer.fit_on_texts(list(x_train) + list(x_test))

x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)

# Preparing weights for model as per evaluation metrics
model_weights = np.ones(len(x_train), dtype=np.float32)
model_weights += train_df[IDENTITY_COLUMNS].sum(axis=1)
model_weights += train_df[TARGET_COLUMN] * (~train_df[IDENTITY_COLUMNS]).sum(axis=1)
model_weights += (~train_df[TARGET_COLUMN]) * train_df[IDENTITY_COLUMNS].sum(axis=1) * 5
model_weights /= model_weights.mean()

def generate_embedding_matrix(word_index, path):
    embedding_index = KeyedVectors.load(path, mmap='r')
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        for candidate in [word, word.lower()]:
            if candidate in embedding_index:
                embedding_matrix[i] = embedding_index[candidate]
                break
    return embedding_matrix
    
def build_model(embedding_matrix, num_aux_targets):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = Dropout(0.1)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x)
    x = Dropout(0.1)(x)
    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x),])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
    result = Dense(1, activation='sigmoid')(hidden)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(hidden)
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
    return model
    
embedding_matrix = np.concatenate([generate_embedding_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

for model_idx in range(NUM_MODELS):
    model = build_model(embedding_matrix, y_aux_train.shape[-1])
    for global_epoch in range(EPOCHS):
        model.fit(x_train,
            [y_train, y_aux_train],
            batch_size=BATCH_SIZE,
            epochs=1,
            verbose=1,
            sample_weight=[model_weights.values, np.ones_like(model_weights)],
        )
        checkpoint_predictions.append(model.predict(x_test, batch_size=128)[0].flatten())
        weights.append(2 ** global_epoch)

predictions = np.average(checkpoint_predictions, weights=weights, axis=0)

submission = pd.DataFrame.from_dict({
    'id': test_df.id,
    'prediction': predictions
})
submission.to_csv('submission.csv', index=False)

model.save('lstm.h5')
model.save_weights('weights.h5')
model.save('lstm.model')