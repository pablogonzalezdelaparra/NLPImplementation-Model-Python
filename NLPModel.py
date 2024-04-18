import re
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.metrics import pairwise
import pprint

from Preprocessor import Preprocessor


class NLPModel:
    def __init__(self):
        pass

    def run(self, train_path, test_path, n):
        # Clean the data
        train_data, train_enum = self.clean_data(train_path)
        test_data, test_enum = self.clean_data(test_path)

        # Create n-grams
        train_n_gram = self.get_ngrams(train_data, n)
        test_n_gram = self.get_ngrams(test_data, n)

        # Flatten the data
        train_n_gram_corpus = self.flatten_data(train_n_gram)

        # One-hot encoding
        mat_train = self.one_hot_encoding(train_n_gram_corpus, train_n_gram)
        mat_test = self.one_hot_encoding(train_n_gram_corpus, test_n_gram)

        # Evaluate the similarity between the two datasets
        similarities = self.evaluate(
            mat_train, mat_test, train_enum, test_enum
        )

        self.compare(similarities)

        return []
    
    def compare(self, similarities):  
        max_similarity = {}
        comparison = {} 
        for text in similarities:
            for row in text:
                if tuple(row[0]) not in max_similarity:
                    max_similarity[tuple(row[0])] = row[2]
                    comparison[tuple(row[0])] = row
                else:
                    if row[2] > max_similarity[tuple(row[0])]:
                        max_similarity[tuple(row[0])] = row[2]
                        comparison[tuple(row[0])] = row
        print("Max similarity")
        pprint.pprint(max_similarity)
        pprint.pprint(comparison)

        
    def evaluate(
        self,
        train_data,
        test_data,
        train_enum,
        test_enum,
    ):
        temp_similarity = []
        similarity_detected = []
        for i in range(len(test_data)):
            for j in range(len(train_data)):
                cosine_similarity = pairwise.cosine_similarity(
                    [test_data[i]], [train_data[j]]
                )
                temp_similarity.append([test_enum[i], train_enum[j], cosine_similarity[0][0]])
            similarity_detected.append(temp_similarity)
        return similarity_detected

    def clean_data(self, folder_path):

        def load_folder(folder_path):
            data = []
            for filename in sorted(os.listdir(folder_path)):
                if filename.endswith(".txt"):
                    with open(
                        os.path.join(folder_path, filename), "r", encoding="utf-8-sig"
                    ) as f:
                        data.append(f.read())
            return data

        temp_data = load_folder(folder_path)
        preprocessor = Preprocessor(temp_data)
        data = preprocessor.clean_data()
        return data

    def get_ngrams(self, data, n):
        n_gram = []

        # TODO: Consider n=1

        for text in data:
            n_gram.append(list(ngrams(text, n)))
        return n_gram

    def flatten_data(self, data):
        return [word for text in data for word in text]

    def one_hot_encoding(self, corpus, data):
        temp_vector = []
        one_hot_test = []

        for sentence in data:
            for word in corpus:
                if word in sentence:
                    temp_vector.append(1)
                else:
                    temp_vector.append(0)
            one_hot_test.append(temp_vector)
            temp_vector = []
        return one_hot_test
