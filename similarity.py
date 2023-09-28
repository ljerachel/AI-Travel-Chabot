from scipy import spatial 
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.util import ngrams
from nltk.stem.snowball import SnowballStemmer


tokenizer = RegexpTokenizer(r"\w+")
english_stopwords = stopwords.words('english') 
# Stemming
sb_stemmer = SnowballStemmer('english')

def cosine_similarity( string_1, string_2):
    strings = [string_1, string_2]

    # Tokenising and removing punctuation
    tokens = []
    for string in strings:
        tokens.append(tokenizer.tokenize(string))

    # Removing stopwords in select dataset only and normalising casing
    docs = []


    for token in tokens:
        docs.append([word.lower() for word in token if word not in english_stopwords])

    # Performing stemming
    stemmed_docs = []
    for doc in docs:
        stemmed_docs.append([sb_stemmer.stem(word) for word in doc])
    docs = stemmed_docs


    # Creating vocabulary
    vocab = []
    for doc in docs:
        for item in doc:
            if item not in vocab:
                vocab.append(item)

    # Creating bag-of-word
    bow = []
    for doc in docs:
        vector = np.zeros(len(vocab))
        for item in doc:
            index = vocab.index(item)
            vector[index] += 1

        bow.append(vector)

    query = bow[0]
    bow = dict(d1=bow[1])
    for d in bow.keys():
        try:
            similarity = 1 - spatial.distance.cosine(query, bow[d])
        except:
            similarity = 0
    return similarity
