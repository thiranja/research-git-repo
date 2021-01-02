from sentence_encording import *

# encording = encodeSentence("I am using bert now for encording")



# print("Hellow")

# print(encording)

sentence1 = "android is the most popular mobile os"
sentence2 = "most of the mobile phones runs with os android"
sentence3 = "here we are testing the semantic similarity"

encording1 = encodeSentence(sentence1)
encording2 = encodeSentence(sentence2)
encording3 = encodeSentence(sentence3)

from scipy import spatial

dist1 = spatial.distance.cosine(encording1.detach(),encording2.detach())
dist2 = spatial.distance.cosine(encording1.detach(),encording3.detach())

print(dist1)
print(dist2)