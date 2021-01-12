from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('paraphrase-distilroberta-base-v1')
# model = SentenceTransformer('stsb-roberta-large')
model = SentenceTransformer('stsb-roberta-base')

def encodeSentence(sentence):
    sentences = [sentence]
    sentence_embedding = model.encode(sentences)
    return sentence_embedding