import gensim

with open('appedit_mobile.txt', 'rb') as txtfile:
	stringLines = txtfile.readlines()
print(stringLines[:3])

# process sentences to tokens
processedLines = [gensim.utils.simple_preprocess(sentence) for sentence in stringLines]

print(processedLines[:3])

model = gensim.models.Word2Vec(processedLines, negative=10, iter=50, min_count=1, size=32)

print(model['android'])

from sklearn.decomposition import PCA
from matplotlib import pyplot

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
