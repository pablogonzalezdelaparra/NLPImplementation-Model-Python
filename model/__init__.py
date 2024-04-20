import os
from nltk.util import ngrams
from sklearn.metrics import pairwise
from model.Preprocessor import Preprocessor


class NLPModel:
    def __init__(self):
        self.__train_n_gram_corpus = []
        self.__train_n_gram = []
        self.__mat_train = []
        self.__train_enum = []
        self.__n_gram = 2

        self.__max_similarity = {}
        self.__average_similarity = {}
        self.__comparison = {}

    def train(self, train_path):
        # Clean the data
        train_data, self.__train_enum = self.__clean_data(train_path)

        # Create n-grams
        self.__train_n_gram = self.__get_ngrams(train_data)

        # Flatten the data
        self.__train_n_gram_corpus = self.__flatten_data(self.__train_n_gram)

        # One-hot encoding
        self.__mat_train = self.__one_hot_encoding(self.__train_n_gram_corpus, self.__train_n_gram)

        return []

    def evaluate(self, test_path):
        # Clean the data
        test_data, test_enum = self.__clean_data(test_path)

        # Create n-grams
        test_n_gram = self.__get_ngrams(test_data)

        # One-hot encoding
        mat_test = self.__one_hot_encoding(self.__train_n_gram_corpus, test_n_gram)

        # Evaluate the similarity between the two datasets
        max_similarity, average_similarity, comparison = self.__compare_texts(
            mat_test, test_enum
        )

        self.__max_similarity = max_similarity
        self.__average_similarity = average_similarity
        self.__comparison = comparison

        return []
    
    def AUC(self, threshold_dic):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for key, value in self.__average_similarity.items():
            if threshold_dic[str(key)] == 1:
                if value > 0.5:
                    TP += 1
                else:
                    FN += 1
            else:
                if value > 0.5:
                    FP += 1
                else:
                    TN += 1

        if TP + FN == 0:
            TPR = 0
        else:
            TPR = TP / (TP + FN)
        if FP + TN == 0:
            FPR = 0
        else:
            FPR = FP / (FP + TN)

        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        AUC = (1 + TPR - FPR) / 2

        print("AUC calculation:")
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, AUC: {AUC}")

        return (TP, FP, TN, FN, AUC)

    def print_max_similarity(self):
        current_key = -1
        print("Max similarity:")
        for key, value in self.__max_similarity.items():
            if key[0] != current_key:
                current_key = key[0]
                print(f"-----Test file #{current_key}-----")
            print(f"Sentence #{int(key[1])+1} --> Train file #{int(value[0][0])+1}. Sentence #{int(value[0][1])+1}: {round((value[1])*100, 3)}%")
        print("\n")

    def print_average_similarity(self):
        print("Average similarity:")
        for key, value in self.__average_similarity.items():
            flag = False
            if value > 0.5:
                flag = True
            print(f"Test file #{int(key)+1} | Plagiarized: {flag} | Average similarity: {round((value)*100, 3)}%")
        print("\n")

    def print_comparison(self):
        current_key = -1
        print("Comparison:")
        for key, value in self.__comparison.items():
            if value[0][1] != current_key:
                current_key = value[0][1]
                print(f"-----Test file #{current_key}-----")
            print(f"Sentence #{int(value[1][1])+1} --> Train file #{int(value[1][0])+1}. Sentence #{int(value[1][1])+1}: {round((value[2])*100, 3)}%")
        print("\n")
 
    def __compare_texts(
        self,
        mat_test,
        test_enum,
    ):
        max_similarity = {}
        comparison = {}
        for i in range(len(mat_test)):
            for j in range(len(self.__mat_train)):
                cosine_similarity = pairwise.cosine_similarity(
                    [mat_test[i]], [self.__mat_train[j]]
                )
                comparison[(i, j)] = [
                    test_enum[i],
                    self.__train_enum[j],
                    cosine_similarity[0][0],
                ]
                if tuple(test_enum[i]) not in max_similarity:
                    max_similarity[tuple(test_enum[i])] = [
                        self.__train_enum[j],
                        cosine_similarity[0][0],
                    ]
                else:
                    if cosine_similarity[0][0] > max_similarity[tuple(test_enum[i])][1]:
                        max_similarity[tuple(test_enum[i])] = [
                            self.__train_enum[j],
                            cosine_similarity[0][0],
                        ]

        temp_average_similarity = {}
        average_similarity = {}
        for key, value in max_similarity.items():
            if key[0] not in temp_average_similarity:
                temp_average_similarity[key[0]] = [value[1]]
            else:
                temp_average_similarity[key[0]].append(value[1])

        for key, value in temp_average_similarity.items():
            average_similarity[key] = sum(value) / len(value)

        return max_similarity, average_similarity, comparison

    def __clean_data(self, folder_path):

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

    def __get_ngrams(self, data):
        n_gram = []

        # TODO: Consider n=1

        for text in data:
            n_gram.append(list(ngrams(text, self.__n_gram)))
        return n_gram

    def __flatten_data(self, data):
        return [word for text in data for word in text]

    def __one_hot_encoding(self, corpus, data):
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
