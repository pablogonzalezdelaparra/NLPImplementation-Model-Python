import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.metrics import pairwise
from Preprocessor import Preprocessor
import pprint

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
        max_similarity, average_similarity, comparison = self.evaluate(
            mat_train, mat_test, train_enum, test_enum
        )

        return []
    
    def results(self, comparison):
        ""
        {(9, 0): [[9, 0], [6, 0], 0.9781564923143908],
        (9, 1): [[9, 1], [6, 1], 0.2886751345948129],
        (9, 2): [[9, 2], [6, 2], 0.7759402897989853],
        (9, 3): [[9, 3], [6, 3], 0.7537783614444091],
        (9, 4): [[9, 4], [6, 4], 0.9952718411247492],
        (9, 5): [[9, 5], [6, 5], 0.6324555320336759]}
        ""

        average_similarity = {}
        for key, value in comparison.items():
            print(f"Text-sentence {key} is most similar to text-sentence {value[1]} with a similarity of {round(value[2], 2)}")
            if key[0] not in average_similarity:
                average_similarity[key[0]] = [value[2]]
            else:
                average_similarity[key[0]].append(value[2])
        for key, value in average_similarity.items():
            print(f"Average similarity for text {key} is {round(sum(value)/len(value), 2)}")
    
    def evaluate(
        self,
        train_data,
        test_data,
        train_enum,
        test_enum,
    ):
        max_similarity = {}
        comparison = {}
        for i in range(len(test_data)):
            for j in range(len(train_data)):
                cosine_similarity = pairwise.cosine_similarity(
                    [test_data[i]], [train_data[j]]
                )
                comparison[(i, j)] = [test_enum[i], train_enum[j], cosine_similarity[0][0]]
                if tuple(test_enum[i]) not in max_similarity:
                    max_similarity[tuple(test_enum[i])] = [train_enum[j], cosine_similarity[0][0]]
                else:
                    if cosine_similarity[0][0] > max_similarity[tuple(test_enum[i])][1]:
                        max_similarity[tuple(test_enum[i])] = [train_enum[j], cosine_similarity[0][0]]

        pprint.pp(max_similarity)
        
        temp_average_similarity = {}
        average_similarity = {}
        for key, value in max_similarity.items():
            if key[0] not in temp_average_similarity:
                temp_average_similarity[key[0]] = [value[1]]
            else:
                temp_average_similarity[key[0]].append(value[1])

        for key, value in temp_average_similarity.items():
            average_similarity[key] = sum(value) / len(value)

        pprint.pp(average_similarity)

        return max_similarity, average_similarity, comparison

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
