from sentence_transformers import SentenceTransformer

# importing the bert model for sentence encording
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# model = SentenceTransformer('stsb-roberta-large')
model = SentenceTransformer('stsb-roberta-base')

# function to encode sentences
def encodeSentence(sentence):
    sentences = [sentence]
    sentence_embedding = model.encode(sentences)
    return sentence_embedding