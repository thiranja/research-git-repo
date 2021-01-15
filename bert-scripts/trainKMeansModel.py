from KMeans import *
from fileManipulation import fimport

# importing the keywords dataset and making the keyword item array

filename = 'keywordset.csv'
keywords = []

df = fimport(filename)

for i in range(0, len(df)):
    keywords.append( KeywordItem(df.loc[i,'Keyword']) )

# Training parameters
noOfCentroids = 200
noOfKeywords = len(keywords)
gapBetweenRandomCentroids = noOfKeywords//noOfCentroids
modelName = "200centroidRobertaBaseModel.json"
dimentionality = 768

# Creating a random list of centroids from dataset itself to inizially load kmeans model

centroidVectorList = []
for i in range(0,noOfCentroids):
    centroidVectorList.append(keywords[gapBetweenRandomCentroids*i].getVector())
    
# Making the kmeans model and training it to keyword set

kmeansModel = KMeans(noOfCentroids,dimentionality,centroidCordinates=centroidVectorList)
kmeansModel.fit(keywords)

# saving the trained model to a json file

kmeansModel.saveModelToAFile(modelName)
