from sbert_sentence_encording import *

# encording = encodeSentence("I am using bert now for encording")

import time

# print("Hellow")

# print(encording)

sentence1 = "android is the most popular mobile os"
sentence2 = "most of the mobile phones runs with os android"
sentence3 = "here we are testing the semantic similarity"
time1 = time.time()
encording1 = encodeSentence(sentence1)
time2 = time.time()
encording2 = encodeSentence(sentence2)
encording3 = encodeSentence(sentence3)

print(encording1)

from scipy import spatial
time3 = time.time()
dist1 = spatial.distance.euclidean(encording1,encording2)
time4 = time.time()
dist2 = spatial.distance.cosine(encording1,encording3)
time5 = time.time()

timea = time4 - time3
timea2 = time5 - time4
timeb = time2 - time1
print(timea)
print(timea2)
print(timeb)
print(timea/timeb)
print(timea2/timeb)
# print(dist1)
# print(dist2)

# import numpy as np
# import pandas as pd

# def fimport(filename,sep=',',encoding='utf-16',skiprows=0, header=0):
# 	frame = pd.DataFrame()
# 	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, error_bad_lines=False)
# 	return frame

# filename = 'keywordset.csv'
# keywords = []

# df = fimport(filename)

# for i in range(0,len(df)):
#     keywords.append( df.loc[i,'Keyword'] )

# keywordEncodeingPairs = []

# from KeywordEncodePair import KewordEncodePair

# for keyword in keywords:
#     encode = encodeSentence(keyword)
#     pair = KewordEncodePair(keyword, encode)
#     # keywordEncodeingPairs.append(pair)

# print(len(keywordEncodeingPairs))


