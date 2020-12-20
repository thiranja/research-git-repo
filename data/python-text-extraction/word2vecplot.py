from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
import nltk
nltk.download('punkt')
from nltk import word_tokenize
# define training data

with open('appedit_mobile.txt', 'rb') as txtfile:
	stringLines = txtfile.readlines()
print(stringLines[:3])

import gensim

# process sentences to tokens
processedLines = [gensim.utils.simple_preprocess(sentence) for sentence in stringLines]
#create word list from token using utf8 encoding 
word_list = [word.encode('utf-8') for words in processedLines for word in words]
 
#check the length of the list
print('Length: ', len(word_list))
#check first five words
print(word_list[:5])

#tokenize text
#tokenized_text = word_tokenize(stringLines)
#print(tokenized_text[:3])

#from sklearn.feature_extraction.text import CountVectorizer
#vectorizer=CountVectorizer()
#data_corpus=["guru99 is the best sitefor online tutorials. I love to visit guru99."]
#data_corpus = txtfile
#vocabulary=vectorizer.fit(data_corpus)
#X= vectorizer.transform(data_corpus)
#print(X.toarray())
#print(vocabulary.get_feature_names())
	
model = gensim.models.Word2Vec([word_list], negative=10, iter=50, min_count=1, size=32)
    
model['uncourteous']
# train model
#model = Word2Vec([word_list], min_count=1)
# fit a 2d PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
pyplot.show()
