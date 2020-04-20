import csv
from csv import DictWriter
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from contractions import contractions_dict
import numpy as np
from tqdm import tqdm, trange
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from wordsegment import load, segment
from pattern.en import suggest
load()
from keras.preprocessing.text import Tokenizer
import pandas as pd
spell = SpellChecker()
# =============================================================================
# from tensorflow.nn import relu, sigmoid
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# =============================================================================
def append_dict_as_row(file_name, dict_of_elem, field_names):
    with open(file_name, 'a+', newline='') as write_obj:     # Open file in append mode
        dict_writer = DictWriter(write_obj, fieldnames=field_names) # Create a writer object from csv module
        dict_writer.writerow(dict_of_elem) # Add dictionary as wor in the csv

def expand_contractions(text, contractions_dict):
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contractions_dict.get(match) \
            if contractions_dict.get(match) \
            else contractions_dict.get(match.lower())
        expanded_contraction = expanded_contraction
        return expanded_contraction

def remove_unnecessary(sentence):
    words = ["u", "r", "ur", "a","b","c","d","e","f","g","h",
             "i","j","k","l","m","n","o","p","q","r","s","t",
             "u","v","w","x","y","z","ur","b4","w8","oh","wr",
             "lol","xD","xP","lolz","lolzzzz","hahahahahaha",
             "haha","ha","hahaha","lolll",":p","okkay","okay",
             "okayyy","okayy","hahahahahahahahhha","â€","€"]
    word_tokens = word_tokenize(sentence) # build tokens to remove unnecessary ones
    string = ''
    for index in range(0, len(word_tokens), 1):
        flag = False
        for itr in range(0, len(words), 1):
            if word_tokens[index] == words[itr]:
                flag = True
                break
        if flag == False:
            if word_tokens[index] not in string:
                string = string + word_tokens[index] + ' '               
    return string

MAX_LEN = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
max_seq_size=256
batch_size = 128
max_words=100000
file_name = "train.csv" 
f = open(file_name, encoding = "utf8")
csv_file = csv.reader(f)
next(csv_file)
comments_column = []
for line in csv_file:
    comments_column.append(line[2])

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1",text)

#nltk.download('punkt')
#nltk.download('stopwords')

input_str = comments_column
dataset_size = len(input_str)
print("Length of Reading file:",dataset_size)
comments = []
stop_words = set(stopwords.words('english')) # Fetching all English Stopwords
for index in range(1700000, dataset_size, 1):
    print("\n*** Sentence #",index,"/",dataset_size,"***")
    input_str[index] = input_str[index].lower()
    input_str[index] = remove_unnecessary(input_str[index])
    input_str[index] = expand_contractions(input_str[index],contractions_dict)
    input_str[index] = reduce_lengthening(input_str[index])
    input_str[index] = " ".join(input_str[index].split())
    #print("\n- Input:",input_str[index])
    clean_sentence = ""
    tokens = word_tokenize(input_str[index])
    misspelled = spell.unknown(tokens)
    for word in misspelled:
        best_word = spell.correction(word) # Get the one `most likely` answer
        for count in range(0, len(tokens), 1):
            if word == tokens[count]:
                tokens[count] = best_word
    #print("tokens:",tokens)
    input_str[index] = ""
    for itr in range (0, len(tokens)):
        input_str[index] += tokens[itr] + ' '
    #print("Input after spellchecker:", input_str[index])
    for k in input_str[index].split("\n"):
        clean_sentence += re.sub(r"[^a-zA-Z0-9]+", ' ', k) # Removing punctuations
        #print("- Clean Sentence (w/o Punctuations):",clean_sentence)
        corrected_tokens = segment(clean_sentence) # Segmentation
        clean_sentence = ""
        for index in range(0, len(corrected_tokens), 1):
            clean_sentence = clean_sentence + corrected_tokens[index] + " " # Converting back to string
        #print("- Segment Tokens:", clean_sentence)
        word_tokens = word_tokenize(clean_sentence) # splitting tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words] # Sentence without English stopwords  
        #print("- Filtered words(w/o Eng-Stopwords):", filtered_sentence)
        lemmatized_sentence = []
        lemmatizer=WordNetLemmatizer()
        for word in filtered_sentence:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        #print("- Lemmatized Sentence:", lemmatized_sentence)
        comments.append(lemmatized_sentence)
#print("\nLemmatized Comments in list:", comments)
comments_strings = []
for index in range(0, len(comments), 1):
    string = ''
    for itr in range(0, len(comments[index]), 1):
        string = string + comments[index][itr] + ' '
    comments_strings.append(string)
#print("Refined Comments:",comments_strings)
dataset = pd.read_csv('train.csv')
field_names = ['id', 'target', 'comments_text', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat', 'asian',	'atheist',
               'bisexual', 'black',	'buddhist',	'christian', 'female',	'heterosexual',	'hindu', 
               'homosexual_gay_or_lesbian',	'intellectual_or_learning_disability', 'jewish', 'latino',
               'male',	'muslim', 'other_disability', 'other_gender', 'other_race_or_ethnicity', 'other_religion',
               'other_sexual_orientation', 'physical_disability', 'psychiatric_or_mental_illness',
               'transgender', 'white', 'created_date', 'publication_id', 'parent_id', 'article_id',
               'rating', 'funny', 'wow', 'sad',	'likes', 'disagree', 'sexual_explicit',
               'identity_annotator_count', 'toxicity_annotator_count']
# =============================================================================
# dict = {
#         'id': 'id', 'target': 'target', 'comments_text': 'comments_text', 'severe_toxicity': 'severe_toxicity',	
#         'obscene': 'obscene', 'identity_attack': 'identity_attack','insult': 'insult', 'threat': 'threat',
#         'asian': 'asian', 'atheist': 'atheist', 'bisexual': 'bisexual', 'black': 'black', 'buddhist': 'buddhist',	
#         'christian': 'christian', 'female': 'female', 'heterosexual': 'heterosexual', 'hindu': 'hindu',
#         'homosexual_gay_or_lesbian': 'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability': 'intellectual_or_learning_disability',
#         'jewish': 'jewish', 'latino': 'latino', 'male': 'male', 'muslim': 'muslim', 'other_disability': 'other_disability',
#         'other_gender': 'other_gender',	'other_race_or_ethnicity': 'other_race_or_ethnicity',	
#         'other_religion': 'other_religion',	'other_sexual_orientation': 'other_sexual_orientation',	
#         'physical_disability': 'physical_disability', 'psychiatric_or_mental_illness': 'psychiatric_or_mental_illness',	
#         'transgender': 'transgender', 'white': 'white', 'created_date': 'created_date', 'publication_id': 'publication_id',	
#         'parent_id': 'parent_id', 'article_id': 'article_id', 'rating': 'rating', 'funny': 'funny', 'wow': 'wow',
#         'sad': 'sad', 'likes': 'likes', 'disagree': 'disagree','sexual_explicit': 'sexual_explicit', 
#         'identity_annotator_count': 'identity_annotator_count', 'toxicity_annotator_count': 'toxicity_annotator_count'
#         }
# append_dict_as_row('refined_train.csv', dict, field_names)
# =============================================================================
for index in range(0, len(comments_strings)):
    print(index+1,"/",len(comments_strings))
    dict = {
        'id': dataset.loc[index,'id'], 'target': dataset.loc[index,'target'],
        'comments_text': comments_strings[index], 'severe_toxicity':dataset.loc[index,'severe_toxicity'],	
        'obscene': dataset.loc[index,'obscene'], 'identity_attack': dataset.loc[index,'identity_attack'],
        'insult': dataset.loc[index,'insult'], 'threat': dataset.loc[index,'threat'],	
        'asian': dataset.loc[index,'asian'], 'atheist': dataset.loc[index,'atheist'],
        'bisexual': dataset.loc[index,'bisexual'], 'black': dataset.loc[index,'black'],	
        'buddhist': dataset.loc[index,'buddhist'], 'christian': dataset.loc[index,'christian'],
        'female': dataset.loc[index,'female'],'heterosexual': dataset.loc[index,'heterosexual'],
        'hindu': dataset.loc[index,'hindu'], 'homosexual_gay_or_lesbian': dataset.loc[index,'homosexual_gay_or_lesbian'],	
        'intellectual_or_learning_disability': dataset.loc[index,'intellectual_or_learning_disability'],
        'jewish': dataset.loc[index,'jewish'], 'latino': dataset.loc[index,'latino'],
        'male': dataset.loc[index,'male'], 'muslim': dataset.loc[index,'muslim'], 
        'other_disability': dataset.loc[index,'other_disability'], 'other_gender': dataset.loc[index,'other_gender'],	
        'other_race_or_ethnicity': dataset.loc[index,'other_race_or_ethnicity'],	
        'other_religion': dataset.loc[index,'other_religion'],	
        'other_sexual_orientation': dataset.loc[index,'other_sexual_orientation'],	
        'physical_disability': dataset.loc[index,'physical_disability'],	
        'psychiatric_or_mental_illness': dataset.loc[index,'psychiatric_or_mental_illness'],	
        'transgender': dataset.loc[index,'transgender'],
        'white': dataset.loc[index,'white'], 'created_date': dataset.loc[index,'created_date'],
        'publication_id': dataset.loc[index,'publication_id'],	
        'parent_id': dataset.loc[index,'parent_id'], 'article_id': dataset.loc[index,'article_id'],
        'rating': dataset.loc[index,'rating'], 'funny': dataset.loc[index,'funny'],
        'wow': dataset.loc[index,'wow'], 'sad': dataset.loc[index,'sad'],
        'likes': dataset.loc[index,'likes'], 'disagree': dataset.loc[index,'disagree'],	
        'sexual_explicit': dataset.loc[index,'sexual_explicit'],	
        'identity_annotator_count': dataset.loc[index,'identity_annotator_count'],	
        'toxicity_annotator_count': dataset.loc[index,'toxicity_annotator_count']
        }
    append_dict_as_row('preprocessed_train.csv', dict, field_names)

new_dataset = pd.read_csv('preprocessed_train.csv')
print("New Dataset size:", len(new_dataset))
# =============================================================================
# tok = Tokenizer()
# tok.fit_on_texts(comments_strings)
# print("Word_counts", tok.word_counts)
# print("\nDocuments_counts", tok.document_count)
# print("\nWord_index", tok.word_index)
# print("\nWord_docs", tok.word_docs)
# 
# # =============================================================================
# # #integer encode documents
# # encoded_docs = tok.texts_to_matrix(comments_strings, mode='count')
# # print("\nEncoded_docs", encoded_docs)
# # =============================================================================
# 
# # =============================================================================
# # train = pd.read_csv('padded_full.csv')
# # x_train = train['comment_text']
# # =============================================================================
# 
# test = pd.read_csv('test.csv')
# x_test = test['comment_text']
# 
# train_sequence = tok.texts_to_sequences(comments_strings)
# print(train_sequence)
# vocab_size = len(train_sequence)+1
# test_sequence = tok.texts_to_sequences(x_test)
# 
# # =============================================================================
# # print("\n After texts_to_sequences")
# # print("train(length):",len(train_sequence[1]))
# # print("test:(length):",len(test_sequence[1]))
# # =============================================================================
# 
# # =============================================================================
# # print("x_train:",x_train)
# # print("x_test:",x_test)
# # =============================================================================
# 
# train_padded = sequence.pad_sequences(train_sequence, maxlen=MAX_LEN, padding=padding_type, truncating=trunc_type)
# test_padded = sequence.pad_sequences(test_sequence, maxlen=MAX_LEN, padding=padding_type, truncating=trunc_type)
# 
# # =============================================================================
# # print("\nAfter pad_sequence")
# # print("train(length():",len(train_padded[1]))
# # print("test(length):",len(test_padded[1]))
# # print("train:",train_padded[1])
# # print("test:",test_padded[1])
# # =============================================================================
# 
# print("\nLoading Glove Model...")
# f = open("glove_files/glove.6B.100d.txt",'r', encoding="utf8") # Load Model
# embedding_matrix = np.zeros((dataset_size+1,100))
# embedding_values = {}
# for line in tqdm(f):
#     value = line.split(' ')
#     word = value[0]
#     coeff = np.array(value[1:],dtype = 'float32')
#     embedding_values[word]=coeff
# print("Glove Model Loaded")
# #print(embedding_values)
# 
# # Preparing Matrix for word Embeddings out of Vocab used
# for word,i in tqdm(tok.word_index.items()):
#     values = embedding_values.get(word)
#     if values is not None:
#         embedding_matrix[i] = values
# 
# print(len(embedding_matrix))
# for index in range(0 , len(embedding_matrix), 1):
#     if index == 10:
#         break
#     print(index,"/",len(embedding_matrix))
#     print(embedding_matrix[index])
# 
# embedding_size = 100
# sequence_input = Input(shape=(max_seq_size,), dtype='int32')
# embedding_layer = Embedding(vocab_size,
#                             embedding_size,
#                             weights=[embedding_matrix],
#                             input_length=max_seq_size,
#                             trainable=False)
# x_layer = embedding_layer(sequence_input)
# x_layer = SpatialDropout1D(0.2)(x_layer)
# x_layer = Bidirectional(CuDNNGRU(64, return_sequences=True))(x_layer)   
# x_layer = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x_layer)
# avg_pool1 = GlobalAveragePooling1D()(x_layer)
# max_pool1 = GlobalMaxPooling1D()(x_layer)     
# x_layer = concatenate([avg_pool1, max_pool1])
# preds = Dense(1, activation=sigmoid)(x_layer)
# model = Model(sequence_input, preds)
# model.summary()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# callbacks = [
#     EarlyStopping(patience=10, verbose=1),
#     ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
#     ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
# ]
# history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=callbacks)
# =============================================================================
