import pandas as pd
import numpy as np

def fimport(filename,sep=',',encoding='utf-16',skiprows=1, header=0):
	frame = pd.DataFrame()
	frame = pd.read_csv(filename,sep=sep, skiprows=skiprows, header=header, encoding=encoding, error_bad_lines=False)
	return frame
	
filenames = ['Android Advices Keyword Stats 2020-10-20 at 01_20_28.csv'
, 'Android Authority Keyword Stats 2020-10-20 at 01_26_08.csv'
, 'Android Central Keyword Stats 2020-10-20 at 01_15_24.csv'
, 'Android Guys Keyword Stats 2020-10-20 at 01_24_13.csv'
, 'Android Headlines Keyword Stats 2020-10-20 at 01_27_46.csv'
, 'Android Police Keyword Stats 2020-10-20 at 01_29_25.csv'
, 'Appedit mobile Keyword Stats 2020-10-18 at 04_24_41.csv'
, 'Best for Android Keyword Stats 2020-10-18 at 04_36_16.csv'
, 'Droid Life Keyword Stats 2020-10-20 at 01_17_48.csv'
, 'Make Use of Keyword Stats 2020-10-20 at 00_44_17.csv'
, 'Mobile internist Keyword Stats 2020-10-18 at 04_34_29.csv'
, 'XDA Forum Keyword Stats 2020-10-18 at 04_28_55.csv'
, 'XDA Keyword Stats 2020-10-18 at 04_26_36.csv']

print(filenames)

uniqueKeywords = []

for fileName in filenames:
	df = fimport(fileName)
	
	length = len(df)
	for i in range(0,  length):
		
		isAlreadyAdded = False
		keyword = df.loc[i,'Keyword']
		
		for j in range(0, len(uniqueKeywords)):
			if uniqueKeywords[j] == keyword:
				isAlreadyAdded = True
		if (isAlreadyAdded):
			continue
		else:
			uniqueKeywords.append(keyword)
			
print(len(uniqueKeywords))

uniqueKeywords.sort()
subSetRemovedKeywords = []

def isSubphrase(subphrase,phrase):
    subphraseArray = subphrase.split(" ")
    phraseArray = phrase.split(" ")
    for word in subphraseArray:
        if word in phraseArray:
            continue
        else:
            return False
    return True

def removeSubstringKeywords(keywords, newKeywords):
    for i in range( 0 , len(keywords)):
        keyword = keywords[i]
        if (len(newKeywords) == 0):
            newKeywords.append(keyword)
            continue
        
        canAppendKeyword = True
        newKeywordsLength = len(newKeywords)
        for j in range ( 0, newKeywordsLength):
            #print(newKeywords)
            newKeyword = newKeywords[j]
            if isSubphrase(keyword,newKeyword):
                canAppendKeyword = False
                break
            if isSubphrase(newKeyword,keyword):
                newKeywords.remove(newKeyword)
                newKeywordsLength -= 1
                break
        if ( canAppendKeyword ):
            newKeywords.append(keyword)

removeSubstringKeywords(uniqueKeywords,subSetRemovedKeywords)

print( len(subSetRemovedKeywords))

# for i in range(0,len(uniqueKeywords) ):
# 	print(uniqueKeywords[i])

# writing keywords into a csv.file
import csv

file = open('unique.csv','w', newline='')

with file:
	header = ['Keyword','Text Phrase']
	writer = csv.DictWriter(file, fieldnames = header)


	writer.writeheader()
	# writing data row wise to the csv file
	for keyword in uniqueKeywords:
		writer.writerow({header[0] : keyword,
						header[1] : 'Not Set'})

file.close()

file = open('uniqueSubSetRemoved.csv','w', newline='')

with file:
	header = ['Keyword','Text Phrase']
	writer = csv.DictWriter(file, fieldnames = header)


	writer.writeheader()
	# writing data row wise to the csv file
	for keyword in subSetRemovedKeywords:
		writer.writerow({header[0] : keyword,
						header[1] : 'Not Set'})

file.close()

file = open('keywordset.csv','w', newline='')

with file:
	header = ['Keyword']
	writer = csv.DictWriter(file, fieldnames = header)

	writer.writeheader()
	# writing data row wise to the csv file
	for keyword in subSetRemovedKeywords:
		writer.writerow({header[0] : keyword})

file.close()