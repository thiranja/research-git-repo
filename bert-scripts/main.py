from KMeans import *

# Loading model from saved file
modelName = "200centroidRobertaBaseModel.json"
kMeansModel = loadModelFromFile(modelName)

# Using model to guess some random phrases 
# # print(model.getMatchingKeyword(""))
print(kMeansModel.getMatchingKeyword("samsung one ui"))
print(kMeansModel.getMatchingKeyword("mobile games with refresh rate 120hz"))
print(kMeansModel.getMatchingKeyword("acr recorder for android pie"))
print(kMeansModel.getMatchingKeyword("flagship phones from year 2019"))
print(kMeansModel.getMatchingKeyword("45w fast charger for samsung phones"))
print(kMeansModel.getMatchingKeyword("getting android root access with pc"))
print(kMeansModel.getMatchingKeyword("adx mobile softwear development community"))
print(kMeansModel.getMatchingKeyword("benifits of the iphone"))
print(kMeansModel.getMatchingKeyword("add boarding pass to google pay app"))
print(kMeansModel.getMatchingKeyword("activate screen call feature in google pixel 3"))
print(kMeansModel.getMatchingKeyword("install google playstore to amazon fire tablet"))
print(kMeansModel.getMatchingKeyword("turn on speed camera alerts in google maps"))
print(kMeansModel.getMatchingKeyword("oppo a37 running on android 6.0"))
print(kMeansModel.getMatchingKeyword("samsung galaxy tab s3 running on android 9 pie"))


