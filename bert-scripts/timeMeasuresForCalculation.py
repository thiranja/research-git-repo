from sbert_sentence_encording import *
import time

# example sentences for getting time measures
sentence1 = "android is the most popular mobile os"
sentence2 = "most of the mobile phones runs with os android"
sentence3 = "here we are testing the semantic similarity"

# measuring time between encording sentence
time1 = time.time()
encording1 = encodeSentence(sentence1)
time2 = time.time()
encording2 = encodeSentence(sentence2)
encording3 = encodeSentence(sentence3)

# using spatial from scipy to get disance measures getting time between them
from scipy import spatial
time3 = time.time()
dist1 = spatial.distance.euclidean(encording1,encording2)
time4 = time.time()
dist2 = spatial.distance.cosine(encording1,encording3)
time5 = time.time()

# calculating and printing the time for encording sentences and distance measures
timeForCalculatingEuclidianDistance = time4 - time3
timeForCalculatingCosineDistance = time5 - time4
timeForEncordingSentence = time2 - time1
print(timeForEncordingSentence)
print(timeForCalculatingEuclidianDistance)
print(timeForCalculatingCosineDistance)

# calculating the distance ratio in the final equation
print((timeForEncordingSentence + timeForCalculatingCosineDistance)/timeForCalculatingCosineDistance)