# keywords = ["This", "this","This is", "This is not","This is the","This is the keyword","This is the","This the"]
# newKeywords = []

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

# isSubphrase(keywords[7],keywords[5])
# removeSubstringKeywords(keywords,newKeywords)

# print(newKeywords)

