"""
Created on Thu Jul 22 23:43:23 2020
@author: faisa
"""
import pandas as pd
import time
import datetime as dt

def calcProcessTime(starttime, cur_iter, max_iter):

    telapsed = time.time() - starttime
    testimated = (telapsed/cur_iter)*(max_iter)

    finishtime = starttime + testimated
    finishtime = dt.datetime.fromtimestamp(finishtime).strftime("%H:%M:%S")  # in time

    lefttime = testimated-telapsed  # in seconds

    return (int(telapsed), int(lefttime))

cur_iter = 0
stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all",
             "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", 
             "be", "because", "been", "before", "being", "below", "between",
             "both", "but", "by", "can", "couldn", "couldn't", "d", "did",
             "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", 
             "don", "don't", "dont", "down", "during", "each", "few", "for", "from",
             "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", 
             "have", "haven", "haven't", "having", "he", "her", "here", "hers",
             "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
             "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "couldnt"
             "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn",
             "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not",
             "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", 
             "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", 
             "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", 
             "so", "some", "such", "t", "than", "that", "that'll", "the", "their", 
             "theirs", "them", "themselves", "then", "there", "these", "they", "this", 
             "those", "through", "to", "too", "under", "until", "up", "ve", "very", 
             "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", 
             "where", "which", "while", "who", "whom", "why", "will", "with", "won", 
             "won't", "wouldn", "wouldn't", "wouldnt","who's", "whos", "theirs", "their's", 
             "y", "you", "you'd", "you'll", "you're", "shouldnt", "hasnt", "havent", "aint", "doesnt",
             "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "didnt"
             "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's",
             "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll",
             "they're", "they've", "we'd", "we'll", "we're", "we've", "what's",
             "when's", "where's", "who's", "why's", "would"]
print("Available Stopwords:", len(stopwords))
dataset = pd.read_csv('preprocessed_train.csv', encoding = 'utf8')
size = len(dataset)
#print(dataset.loc[0, 'comments_text'])
#string = dataset.loc[0, 'comments_text']
#print(string)
#dataset.loc[0, 'comments_text'] = "Hahahahaaha changed"
#print(dataset.loc[0, 'comments_text'])
start = time.time()
for index in range(0, size, 1):
    print("Preprocessing ...", index+1, "/", size)
    new_comment = ""
    comment = dataset.loc[index, 'comments_text']
    #print("Original comment:", comment)
    if(comment == '' or pd.isnull(comment)):
        print("Skipping the empty string")
    else:
        tokens = comment.split()
        for chunk in tokens:
            if chunk not in stopwords:
                new_comment += chunk + " "
        #print("New Comment:", new_comment,"\n")
        dataset.loc[index, 'comments_text'] = new_comment
        cur_iter += 1
        prstime = calcProcessTime(start,cur_iter, size)
        print("time elapsed: %s(s), time left: %s(s)"%prstime)
        #time.sleep(0.5)

dataset.to_csv('testing_output.csv')