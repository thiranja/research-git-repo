from random import random
import numpy as np
from scipy import spatial
import json
from sbert_sentence_encording import encodeSentence

# String to return when no keyword is matching
NO_KEYWORD_MATCHING_STRING = "No Keyword Matching"
SEMANTIC_DISTANCE_THRESHOLD = 15.0

# This function is used to load a previouly saved model as a json file
def loadModelFromFile(filename):
    # opening the target json file
    with open(filename) as json_file:
        model = json.load(json_file)
        # fetching the information about kmeans model
        dimetionality = model['dimetionality']
        noOfCentroids = model['noOfCentroids']
        centroidsObject = model['centroids']
        # Creating a list for centroids and iterating over centroid item in json file
        centroids = []
        for centroidItem in centroidsObject:
            # Fetch the details of centroid
            id = centroidItem['id']
            dimetionality = centroidItem['dimetionality']
            cordinatesArray = centroidItem['cordinates']
            itemsArray = centroidItem['items']
            # creating a cordinate vector inizaially with zeros
            cordinates = np.zeros(shape=(1,dimetionality))
            # Assigning cordinate values iterating through the cordinates
            for i in range(0,len(cordinatesArray)):
                cordinates[0][i] = np.float32(cordinatesArray[i])
            items = []
            # Assigning items values iterating through the items
            for i in range(0,len(itemsArray)):
                item = KeywordItem(itemsArray[i])
                items.append(item)
            # Creating centroid object from fetched data and appending to centroid list
            centroid = Centroid(dimetionality, id, cordinates)
            centroid.setItems(items)
            centroids.append(centroid)
        #  Creating and returning the kmeans model from centroid list and fetched data
        kmeansModel = KMeans(noOfCentroids,dimetionality)
        kmeansModel.setCentroids(centroids)
        return kmeansModel

# This function is for identifing the index of minimum value of a python list
# This approch saves time by only iterating once at array, instead of arr.index(min(arr)) 
def indexOfMinimum(listObj):
    # If list if empty return None
    if len(listObj) == 0:
        return None
    # Else iterate and find the min and index of min and return index
    else:
        min = listObj[0]
        index = 0
        for i in range(1,len(listObj)):
            if (listObj[i] < min):
                min = listObj[i]
                index = i
        return index

# Kmeans class
class KMeans:

    # Constructor of the kmeans class
    def __init__(self,noOfCentroids, dimentionality, centroidCordinates = None):
        self.dimetionality = dimentionality
        self.noOfCentroids = noOfCentroids
        self.centroids = []
        # if centroid cordinates are not given create centroids with random inizial values
        if (centroidCordinates == None):
            for i in range(0,self.noOfCentroids):
                centroid = Centroid(dimentionality, i)
                self.centroids.append(centroid)
        # if centroid cordinates are given create centroids with given cordinates
        else:
            for i in range(0,self.noOfCentroids):
                centroid = Centroid(dimentionality, i,centroidCordinates[i])
                self.centroids.append(centroid)
        
    # This function is used when loading the model from a file to manually set centroids
    def setCentroids(self,centroids):
        self.centroids = centroids

    # Adding items to the correct centroid with the least distance
    def addItemToCentroid(self,item):
        distances = []
        itemVector = item.getVector()
        # Calculating the distance with each centroid
        for centroid in self.centroids:
            centroidVector = centroid.getDimenVector()
            distance = spatial.distance.euclidean(itemVector, centroidVector)
            distances.append(distance)
        # Getting the id of neighrest centroid and adding the item to that centroid
        indexOfMin = indexOfMinimum(distances)
        self.centroids[indexOfMin].addItem(item)
    
    # method to train the dataset on the kmeans model
    def fit(self,items):
        isCentroidsAdjustementFinished = True
        # Run the loop until clusters converge and break the loop
        while True:
            # At each iteration inizialy set the adjustment finish to true
            isCentroidsAdjustementFinished = True
            # Add all the items to respective centrooids
            for item in items:
                self.addItemToCentroid(item)
            # After adding the items adjust the centroids to mean of items
            for centroid in self.centroids:
                temp = centroid.adjustCentroidCordinates()
                if (not temp):
                    # if any of the centroid within the iteration is not in the mean position set adjustmentFinished to false
                    isCentroidsAdjustementFinished = False
            if ( isCentroidsAdjustementFinished):
                break
            # Setting the items in the centroids to empty array again at the end of iteration
            for centroid in self.centroids:
                centroid.resetItems()

    # Experimental method to print the architecture of the kmeans model to verify
    def printModel(self):
        print("***** Printing the Model *****")
        print("No of centroids " , self.noOfCentroids)
        # Iterating over all centroids and printing centroid details
        for centroid in self.centroids:
            print("Centroid ", centroid.id)
            print("Elements in centroid")
            for item in centroid.items:
                print(item.getLabel())
            print("")
    
    # method to save the trained model as a json file
    def saveModelToAFile(self,filename):
        # create a dictionary object and adding model details to it
        model = {}
        model['dimetionality'] = self.dimetionality
        model['noOfCentroids'] = self.noOfCentroids
        model['centroids'] = []
        # make a list for centroids and adding each centroid as a new dic object to the list
        for centroidItem in self.centroids:
            centroid = {}
            centroid['id'] = centroidItem.id
            centroid['dimetionality'] = centroidItem.dimetionality
            centroid['cordinates'] = []
            for value in centroidItem.cordinates[0]:
                centroid['cordinates'].append(str(value))
            centroid['items'] = []
            for item in centroidItem.items:
                centroid['items'].append(item.getLabel())
            model['centroids'].append(centroid)
        # Dumping the dic as a json file to the file
        with open(filename,'w') as output:
            json.dump(model,output)

    # Method to getting the maching keyword from the model
    def getMatchingKeyword(self,phrase):
        encording = encodeSentence(phrase)
        # getting the winiing centroid
        centroidDistances = []
        for centroid in self.centroids:
            centroidVector = centroid.getDimenVector()
            distance = spatial.distance.euclidean(encording, centroidVector)
            centroidDistances.append(distance)
        winnerCentroidId = indexOfMinimum(centroidDistances)
        winnerCentroid = self.centroids[winnerCentroidId]

        # getting the most simillar keyword item from the centroid
        keywordDistances = []
        for keywordItem in winnerCentroid.items:
            itemVector = keywordItem.getVector()
            distance = spatial.distance.euclidean(encording, itemVector)
            keywordDistances.append(distance)
        winnerKeywordId = indexOfMinimum(keywordDistances)
        winnerKeyword = winnerCentroid.items[winnerKeywordId]

        # Semantic similarity threshold this value is used to make sure weather the matching keyword is confident enogough to say similar
        if (keywordDistances[winnerKeywordId] < SEMANTIC_DISTANCE_THRESHOLD):
            return winnerKeyword.getLabel()
        return NO_KEYWORD_MATCHING_STRING
    
    # getting the whole list of keyword suggesions
    def getMatchingKeywordSuggesions(self, phrase):
        encording = encodeSentence(phrase)
        # getting the winiing centroid
        centroidDistances = []
        for centroid in self.centroids:
            centroidVector = centroid.getDimenVector()
            distance = spatial.distance.euclidean(encording, centroidVector)
            centroidDistances.append(distance)
        winnerCentroidId = indexOfMinimum(centroidDistances)
        winnerCentroid = self.centroids[winnerCentroidId]

        # create an array for store matching keyword suggesions
        matchingKeywordSuggesions = []

        # getting the most simillar keyword item from the centroid
        keywordDistances = []
        for keywordItem in winnerCentroid.items:
            itemVector = keywordItem.getVector()
            distance = spatial.distance.euclidean(encording, itemVector)
            # adding keywords for suggesions if there exist keywords that below the semantic distance threshold
            if (distance < SEMANTIC_DISTANCE_THRESHOLD):
                matchingKeywordSuggesions.append(keywordItem.getLabel())
            keywordDistances.append(distance)
        return matchingKeywordSuggesions

    # method to evaluate the model with the test dataset
    def evaluate(self, x, y):
        correctGuesses = 0;
        TotalNumber = len(x)

        print("%-60s %-60s %-60s" %("Text Phrase","Model Guesed Keyword","Keyword"))
        for i in range(0, len(x)):
            textPhrase = x[i]
            trueKeyword = y[i]
            guesedKeyword = self.getMatchingKeyword(textPhrase)
            if (guesedKeyword == trueKeyword):
                correctGuesses = correctGuesses +1
            print("%-60s %-60s %-60s" %(textPhrase, guesedKeyword, trueKeyword))
        accuracy = (correctGuesses/TotalNumber)*100
        print("Accuracy ",accuracy,"%")
        
# Centroid class
class Centroid:

    # constructor of the centroid class
    def __init__(self, dimentionality, id, centroidCordinate = None):
        self.id = id
        self.dimetionality = dimentionality
        # If no inizial cordinates are given the random cordinates will be genarated to the centroid
        if centroidCordinate is None :
            self.cordinates = np.random.rand(1,self.dimetionality)
        # If cordinates are given then set those cordinates as the centroid cordinates
        else:
            self.cordinates = centroidCordinate
        self.items = []
        
    # used to set items manually when loading model from file
    def setItems(self,items):
        self.items = items 

    # Adding a item to centroid
    def addItem(self, item):
        self.items.append(item)

    # method to query the dimention vector of a centroid
    def getDimenVector(self):
        return self.cordinates

    # method to adjust the dimentions of the centroid accorting to the items in the centroid
    # returns a boolian value weather the centroid is adjusted or not
    def adjustCentroidCordinates(self):
        # first create a vector for new cordinates inizialize with zeros
        numOfItems = len(self.items)
        newCordinates = np.zeros(shape=(1,self.dimetionality))
        
        # get the number of items as the divider
        devider = np.array(numOfItems)
        for item in self.items:
            # get the vector of each item
            vector = item.getVector()
            # divide it by devider to get the value it contribute to mean
            meanedVector = vector / devider
            # add the mean value to new cordinate vector
            newCordinates = np.add(newCordinates,meanedVector)
        # check weather the new vector is similar to the already exisiting vector
        isArraySimilar = np.allclose(self.cordinates,newCordinates)
        print(isArraySimilar)
        if (isArraySimilar): 
            # if similar return adjustment finished
            return True
        else:
            # if not similar assign the new cordinates as the cordinates of the centroid and return adjustment is not finished
            self.cordinates = newCordinates
            return False
    
    # reset the items to a empty list used in the training process
    def resetItems(self):
        self.items = []

# Keyword item class keyword items objects are used to hold keyword items with in the model
class KeywordItem:

    # constructor of the class
    def __init__(self,keyword):
        self.lable = keyword

    # return the keyword of the object
    def getLabel(self):
        return self.lable
    
    # return the vector representation of the keyword by encording it by bert model
    def getVector(self):
        encording = encodeSentence(self.getLabel())
        return encording
