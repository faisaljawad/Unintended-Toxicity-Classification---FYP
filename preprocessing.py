# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 13:32:41 2020

@author: faisa
"""

import csv
from csv import DictWriter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from contractions import contractions_dict
import numpy as np
from tqdm import tqdm, trange
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
import pandas as pd
spell = SpellChecker()

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

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1",text)

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english')) # Fetching all English Stopwords
def preprocess_func(input_str):
    print("\nPreprocessing...")
    input_str = input_str.lower()
    print("\n-Input:",input_str)
    input_str = remove_unnecessary(input_str)
    input_str= expand_contractions(input_str,contractions_dict)
    input_str = reduce_lengthening(input_str)
    input_str = " ".join(input_str.split())
    clean_sentence = ""
    tokens = word_tokenize(input_str)
    misspelled = spell.unknown(tokens)
    for word in misspelled:
        best_word = spell.correction(word) # Get the one `most likely` answer
        for count in range(0, len(tokens), 1):
            if word == tokens[count]:
                tokens[count] = best_word
    #print("tokens:",tokens)
    input_str = ""
    for itr in range (0, len(tokens)):
        input_str += tokens[itr] + ' '
    #print("Input after spellchecker:", input_str[index])
    for k in input_str.split("\n"):
        clean_sentence += re.sub(r"[^a-zA-Z0-9]+", ' ', k) # Removing punctuations
        word_tokens = word_tokenize(clean_sentence) # splitting tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words] # Sentence without English stopwords  
        #print("- Filtered words(w/o Eng-Stopwords):", filtered_sentence)
        lemmatized_sentence = []
        lemmatizer=WordNetLemmatizer()
        for word in filtered_sentence:
            lemmatized_sentence.append(lemmatizer.lemmatize(word))
        string = ''
        for index in range(0, len(lemmatized_sentence)):
        	string = string + lemmatized_sentence[index] + ' '
       	return string

