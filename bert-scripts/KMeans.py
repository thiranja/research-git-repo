from random import random
import numpy as np
from scipy import spatial

class KMeans:

    def __init__(self,noOfCentroids, dimentionality):
        self.dimetionality = dimentionality
        self.noOfCentroids = noOfCentroids
        self.centroids = []
        for i in range(0,dimentionality):
            centroid = Centroid(dimentionality, i)
            self.centroids.append(centroid)

    def addItemToCentroid(self,item):
        distances = []
        itemVector = item.getVector()
        for centroid in self.centroids:
            centroidVector = centroid.getDimenVector()
            distance = spatial.distance.euclidean(itemVector, centroidVector)
            distances.append(distance)
        indexOfMin = distances.index(min(distances))
        self.centroids[indexOfMin].addItem(item)
    
    def fit(self,items):
        isCentroidsAdjustementFinished = True
        while True:
            isCentroidsAdjustementFinished = True
            for item in items:
                self.addItemToCentroid(item)
            for centroid in self.centroids:
                temp = centroid.adjustCentroidCordinates()
                if (not temp):
                    isCentroidsAdjustementFinished = False
            if ( isCentroidsAdjustementFinished):
                break
            for centroid in self.centroids:
                centroid.resetItems()

class Centroid:

    def __init__(self, dimentionality, id):
        self.id = id
        self.dimetionality = dimentionality
        self.cordinates = np.random.rand(1,self.dimetionality)
        self.items = []
        

    def addItem(self, item):
        self.items.append(item)

    def getDimenVector(self):
        return self.cordinates

    def adjustCentroidCordinates(self):
        numOfItems = len(self.items)
        newCordinates = np.zeros(shape=(1,self.dimetionality))
        
        devider = np.array(numOfItems)
        for item in self.items:
            vector = item.getVector()
            meanedVector = vector / devider
            newCordinates = np.add(newCordinates,meanedVector)
        isArraySimilar = np.allclose(self.cordinates,newCordinates)
        print(isArraySimilar)
        if (isArraySimilar): 
            return True
        else:
            self.cordinates = newCordinates
            return False
    
    def resetItems(self):
        self.items = []

class Item:

    def __init__(self, lable, vector):
        self.lable = lable
        self.vector = vector

    def getVector(self):
        return self.vector

    def getLabel(self):
        return self.lable
from sentence_encording import encodeSentence

class KeywordItem:

    def __init__(self,keyword):
        self.lable = keyword

    def getLabel(self):
        return self.lable
    
    def getVector(self):
        encording = encodeSentence(self.getLabel())
        return encording.detach()

import pandas as pd

def fimport(filename,sep=',',encoding='utf-16',skiprows=0, header=0):
	frame = pd.DataFrame()
	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, error_bad_lines=False)
	return frame

filename = 'keywordset.csv'
keywords = []

df = fimport(filename)

for i in range(0,len(df)):
    keywords.append( KeywordItem(df.loc[i,'Keyword']) )

print(keywords[0].getVector())

kmeansModel = KMeans(3,768)
kmeansModel.fit(keywords)
# centro = Centroid(100,0)

# for dimen in centro.cordinates:
#     print(dimen)





# items = [
#     Item("A",np.array([0.2,0.3])),
#     Item("B",np.array([0.5,0.1])),
#     Item("C",np.array([0.5,0.5])),
#     Item("D",np.array([0.6,0.9])),
#     Item("E",np.array([0.8,0.7])),
#     Item("F",np.array([1.0,0.8]))
# ]

# model = KMeans(2,2)

# model.fit(items)

# cetroids = model.centroids

# centroid1 = cetroids[0]

# print("Centroid 1")

# print("Centroid Cordinates ")
# print(centroid1.cordinates)
# print("")

# for item in centroid1.items:
#     print(item.lable)

# centroid2 = cetroids[1]

# print("Centroid 2")

# print("Centroid Cordinates ")
# print(centroid2.cordinates)
# print("")

# for item in centroid2.items:
#     print(item.lable)