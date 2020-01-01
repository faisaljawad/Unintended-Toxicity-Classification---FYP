import csv
#import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
from wordsegment import load, segment
from pattern.en import suggest
from sklearn.feature_extraction.text import CountVectorizer

load()
file_name = "first_100.csv" 
f = open(file_name)
csv_file = csv.reader(f)
next(csv_file)
comments_column = []
for line in csv_file:
    comments_column.append(line[2])
#print(comments_column)

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1",text)

#nltk.download('punkt')
#nltk.download('stopwords')

input_str = comments_column
dataset_size = len(input_str)
print("Length of Reading file:",dataset_size)
comments = []
for index in range(0, dataset_size, 1):
    print("\n\n*** Sentence #",index+1,"***")
    input_str[index] = input_str[index].lower()
    input_str[index] = reduce_lengthening(input_str[index])
    print("\n- Input:",input_str[index])
    clean_sentence = ""
    for k in input_str[index].split("\n"):
        clean_sentence = clean_sentence + re.sub(r"[^a-zA-Z0-9]+", ' ', k) # Removing punctuations
        print("- Clean Sentence (w/o Punctuations):",clean_sentence)
        stop_words = set(stopwords.words('english')) # Fetching all English Stopwords
        corrected_tokens = segment(clean_sentence) # Segmentation
        clean_sentence = ""
        for index in range(0, len(corrected_tokens), 1):
            clean_sentence = clean_sentence + corrected_tokens[index] + " " # Converting back to string
        print("- Segment Tokens:", clean_sentence)
        word_tokens = word_tokenize(clean_sentence) # splitting tokens
        filtered_sentence = [w for w in word_tokens if not w in stop_words] # Sentence without English stopwords  
        print("- Filtered words(w/o Eng-Stopwords):", filtered_sentence)
        filtered_words = []
        for w in word_tokens:
            if w not in stop_words:
                correct_word = suggest(w) # Correcting word to nearest possible suggestion
                if len(correct_word) == 1: # If only one suggestion is returned
                    filtered_words.append(correct_word[0][0])
                else: # if multiple suggestions are returned, then choose the one with highest probability
                    maximum = location = 0
                    for index in range(0,len(correct_word),1):
                        if(maximum<correct_word[index][1]):
                            maximum = correct_word[index][1]
                            location = index
                    filtered_words.append(correct_word[location][0])
        filtered_sentence = ""
        for index in range(0, len(filtered_words), 1): 
            filtered_sentence = filtered_sentence + filtered_words[index] + " "
        print("- Filtered Sentence after Words Correction:", filtered_sentence)
        lemmatized_sentence = ""
        lemmatizer=WordNetLemmatizer()
        for word in filtered_sentence:
            lemmatized_sentence = lemmatized_sentence + lemmatizer.lemmatize(word)
        print("- Lemmatized Sentence:", lemmatized_sentence)
        comments.append(lemmatized_sentence)

print("\nLemmatized Comments in list:", comments)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(comments)
print("\nBag of Words:",vectorizer.get_feature_names())
print("\n",X.toarray())
        
        
# =============================================================================
# input_str = [str.lower() for str in input_str]
# input_str = reduce_lengthening(input_str) # Reducing repeated letter till length = 2
# clean_sentence = ""
# for k in input_str.split("\n"):
#     clean_sentence = clean_sentence + re.sub(r"[^a-zA-Z0-9]+", ' ', k) # removing punctuations
# stop_words = set(stopwords.words('english')) # removing stopwords
# corrected_tokens = segment(clean_sentence) # Segmentation
# clean_sentence = ""
# for index in range(0, len(corrected_tokens), 1): 
#     clean_sentence = clean_sentence + corrected_tokens[index] +" " # Converting back to string
# word_tokens = word_tokenize(clean_sentence) # splitting tokens
# filtered_sentence = [w for w in word_tokens if not w in stop_words] # Sentence without stopwords  
# filtered_sentence = [] 
# for w in word_tokens: 
#    if w not in stop_words:
#       correct_word = suggest(w) # Correcting word to nearest possible suggestion
#       if len(correct_word) == 1: # If only one suggestion is returned
#           filtered_sentence.append(correct_word[0][0])
#       else: # if multiple suggestions are returned, then choose the one with highest probability
#            maximum = location = 0
#            for index in range(0,len(correct_word),1):
#                if(maximum<correct_word[index][1]):
#                    maximum = correct_word[index][1]
#                    location = index
#            filtered_sentence.append(correct_word[location][0])
# 
# input_s = ['we ARE all should preach peace', 'we are all good people']
# print(input_s)
# input_s = map(str.lower, input_s) 
# print(input_s)
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(input_str)
# print(vectorizer.get_feature_names())
# print(X.toarray())
# #print("Word Tokens: ",word_tokens,"\n")
# #print("Filtered Sentence: ",filtered_sentence)
# #lemmatizer=WordNetLemmatizer()
# #for word in filtered_sentence:
#     #print(lemmatizer.lemmatize(word))
# =============================================================================
