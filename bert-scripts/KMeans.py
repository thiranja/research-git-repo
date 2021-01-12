from random import random
import numpy as np
from scipy import spatial
import json
from sbert_sentence_encording import encodeSentence

def loadModelFromFile(filename):
    with open(filename) as json_file:
        model = json.load(json_file)
        dimetionality = model['dimetionality']
        # print(dimetionality)
        noOfCentroids = model['noOfCentroids']
        # print(noOfCentroids)
        centroidsObject = model['centroids']
        centroids = []
        for centroidItem in centroidsObject:
            id = centroidItem['id']
            # print(id)
            dimetionality = centroidItem['dimetionality']
            # print(dimetionality)
            cordinatesArray = centroidItem['cordinates']
            itemsArray = centroidItem['items']
            cordinates = np.zeros(shape=(1,dimetionality))
            for i in range(0,len(cordinatesArray)):
                cordinates[0][i] = np.float32(cordinatesArray[i])
            # print(cordinates)
            items = []
            for i in range(0,len(itemsArray)):
                # print(itemsArray[i])
                item = KeywordItem(itemsArray[i])
                items.append(item)
            centroid = Centroid(dimetionality, id, cordinates)
            centroid.setItems(items)
            centroids.append(centroid)
        kmeansModel = KMeans(noOfCentroids,dimetionality)
        kmeansModel.setCentroids(centroids)
        return kmeansModel

class KMeans:

    def __init__(self,noOfCentroids, dimentionality, centroidCordinates = None):
        self.dimetionality = dimentionality
        self.noOfCentroids = noOfCentroids
        self.centroids = []
        if (centroidCordinates == None):
            for i in range(0,self.noOfCentroids):
                centroid = Centroid(dimentionality, i)
                self.centroids.append(centroid)
        else:
            for i in range(0,self.noOfCentroids):
                centroid = Centroid(dimentionality, i,centroidCordinates[i])
                self.centroids.append(centroid)
        

    def setCentroids(self,centroids):
        self.centroids = centroids

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

    def printModel(self):
        print("***** Printing the Model *****")
        print("No of centroids " , self.noOfCentroids)
        for centroid in self.centroids:
            print("Centroid ", centroid.id)
            print("Elements in centroid")
            for item in centroid.items:
                print(item.getLabel())
            print("")
    
    def saveModelToAFile(self,filename):
        model = {}
        model['dimetionality'] = self.dimetionality
        model['noOfCentroids'] = self.noOfCentroids
        model['centroids'] = []
        for centroidItem in self.centroids:
            centroid = {}
            centroid['id'] = centroidItem.id
            centroid['dimetionality'] = centroidItem.dimetionality
            centroid['cordinates'] = []
            for value in centroidItem.cordinates[0]:
                # print("Printing value")
                # print(value)
                # print("value printed")
                centroid['cordinates'].append(str(value))
            centroid['items'] = []
            for item in centroidItem.items:
                # print(item.getLabel())
                centroid['items'].append(item.getLabel())
            model['centroids'].append(centroid)
        with open(filename,'w') as output:
            json.dump(model,output)

    def getMatchingKeyword(self,phrase):
        encording = encodeSentence(phrase)
        # getting the winiing centroid
        centroidDistances = []
        for centroid in self.centroids:
            centroidVector = centroid.getDimenVector()
            distance = spatial.distance.euclidean(encording, centroidVector)
            centroidDistances.append(distance)
        winnerCentroidId = centroidDistances.index(min(centroidDistances))
        winnerCentroid = self.centroids[winnerCentroidId]

        # getting the most simillar keyword item from the centroid
        keywordDistances = []
        for keywordItem in winnerCentroid.items:
            itemVector = keywordItem.getVector()
            distance = spatial.distance.euclidean(encording, itemVector)
            keywordDistances.append(distance)
        winnerKeywordId = keywordDistances.index(min(keywordDistances))
        winnerKeyword = winnerCentroid.items[winnerKeywordId]

        return winnerKeyword.getLabel()
        




class Centroid:

    def __init__(self, dimentionality, id, centroidCordinate = None):
        self.id = id
        self.dimetionality = dimentionality
        if centroidCordinate is None :
            self.cordinates = np.random.rand(1,self.dimetionality)
        else:
            self.cordinates = centroidCordinate
        self.items = []
        
    def setItems(self,items):
        self.items = items 

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

    def __init__(self, lable, vector= None):
        self.lable = lable
        self.vector = vector

    def getVector(self):
        return self.vector

    def getLabel(self):
        return self.lable

class KeywordItem:

    def __init__(self,keyword):
        self.lable = keyword

    def getLabel(self):
        return self.lable
    
    def getVector(self):
        encording = encodeSentence(self.getLabel())
        return encording

# import pandas as pd

# def fimport(filename,sep=',',encoding='utf-16',skiprows=0, header=0):
# 	frame = pd.DataFrame()
# 	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, error_bad_lines=False)
# 	return frame

# filename = 'keywordset.csv'
# keywords = []

# df = fimport(filename)

# for i in range(0, len(df)):
#     keywords.append( KeywordItem(df.loc[i,'Keyword']) )

# centroidList = []
# for i in range(0,200):
#     centroidList.append(keywords[20*i].getVector())
    

# kmeansModel = KMeans(200,768,centroidCordinates=centroidList)
# kmeansModel.fit(keywords)

# kmeansModel.saveModelToAFile('200centroidRobertaBaseModel.json')

loadedModel = loadModelFromFile('200centroidRobertaBaseModel.json')
loadedModel.printModel()

# model = loadModelFromFile('6dimenModel.json')

# # print(model.getMatchingKeyword(""))
print(loadedModel.getMatchingKeyword("samsung one ui"))
print(loadedModel.getMatchingKeyword("mobile games with refresh rate 120hz"))
print(loadedModel.getMatchingKeyword("acr recorder for android pie"))
print(loadedModel.getMatchingKeyword("flagship phones from year 2019"))
print(loadedModel.getMatchingKeyword("45w fast charger for samsung phones"))
print(loadedModel.getMatchingKeyword("getting android root access with pc"))
print(loadedModel.getMatchingKeyword("adx mobile softwear development community"))
print(loadedModel.getMatchingKeyword("benifits of the iphone"))
print(loadedModel.getMatchingKeyword("add boarding pass to google pay app"))
print(loadedModel.getMatchingKeyword("activate screen call feature in google pixel 3"))
print(loadedModel.getMatchingKeyword("install google playstore to amazon fire tablet"))
print(loadedModel.getMatchingKeyword("turn on speed camera alerts in google maps"))
print(loadedModel.getMatchingKeyword("oppo a37 running on android 6.0"))
print(loadedModel.getMatchingKeyword("samsung galaxy tab s3 running on android 9 pie"))


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

# model.saveModelToAFile('test.json')

# kModel = loadModelFromFile('test.json')

# kModel.printModel()
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