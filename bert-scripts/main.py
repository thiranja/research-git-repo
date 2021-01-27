from KMeans import *
from fileManipulation import fimport

filename = "demo_testset.csv"

df = fimport(filename)

x = []
y = []

for i in range(0, len(df)):
    y.append(df.loc[i,"Keyword"])
    x.append(df.loc[i,"Text Phrase"])

print(len(x))
print(len(y))

print(x[0])
print(y[0])



# Loading model from saved files
modelName = "200centroidRobertaBaseModel.json"
kMeansModel = loadModelFromFile(modelName)
kMeansModel.evaluate(x,y,True)

# textPhrases = [
#     "samsung one ui",
#     "mobile games with refresh rate 120hz",
#     "acr recorder for android pie",
#     "flagship phones from year 2019",
#     "45w fast charger for samsung phones",
#     "getting android root access with pc",
#     "adx mobile softwear development community",
#     "benifits of the iphone",
#     "add boarding pass to google pay app",
#     "activate screen call feature in google pixel 3",
#     "install google playstore to amazon fire tablet",
#     "turn on speed camera alerts in google maps",
#     "oppo a37 running on android 6.0",
#     "samsung galaxy tab s3 running on android 9 pie",
#     "I am trying to read a book",
#     "Doctor went to the hospital",
#     "pass the beer",
#     "apps always move back to the internal storage",
#     "review on asus max pro m2's camera"
# ]

# for phrase in textPhrases:
#     print("Text Phrase      ",phrase)
#     print("-----------------------------")
#     suggestions = kMeansModel.getMatchingKeywordSuggesions(phrase)
#     print("Matching Keywords")
#     for suggetion in suggestions:
#         print("             ",suggetion)
#     print("-----------------------------")
#     print()

# Using model to guess some random phrases 
# # print(model.getMatchingKeyword(""))
# print(kMeansModel.getMatchingKeyword("samsung one ui"))
# print(kMeansModel.getMatchingKeyword("mobile games with refresh rate 120hz"))
# print(kMeansModel.getMatchingKeyword("acr recorder for android pie"))
# print(kMeansModel.getMatchingKeyword("flagship phones from year 2019"))
# print(kMeansModel.getMatchingKeyword("45w fast charger for samsung phones"))
# print(kMeansModel.getMatchingKeyword("getting android root access with pc"))
# print(kMeansModel.getMatchingKeyword("adx mobile softwear development community"))
# print(kMeansModel.getMatchingKeyword("benifits of the iphone"))
# print(kMeansModel.getMatchingKeyword("add boarding pass to google pay app"))
# print(kMeansModel.getMatchingKeyword("activate screen call feature in google pixel 3"))
# print(kMeansModel.getMatchingKeyword("install google playstore to amazon fire tablet"))
# print(kMeansModel.getMatchingKeyword("turn on speed camera alerts in google maps"))
# print(kMeansModel.getMatchingKeyword("oppo a37 running on android 6.0"))
# print(kMeansModel.getMatchingKeyword("samsung galaxy tab s3 running on android 9 pie"))
# print(kMeansModel.getMatchingKeyword("I am trying to read a book"))
# print(kMeansModel.getMatchingKeyword("Doctor went to the hospital"))
# print(kMeansModel.getMatchingKeyword("pass the beer"))


