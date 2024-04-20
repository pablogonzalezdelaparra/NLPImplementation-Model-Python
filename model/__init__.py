import os
from nltk.util import ngrams
from sklearn.metrics import pairwise
from model.Preprocessor import Preprocessor


class NLPModel:
    """
    The NLPModel class is used to train and evaluate the model.
    """

    def __init__(self, similarity_limit=0.5, n_gram=2):
        """
        Initialize the model with the necessary parameters.
        """
        self.__similarity_limit = similarity_limit
        self.__n_gram = n_gram

    def prepare_text(self, folder_path):
        """
        Prepare the text data from the given folder path.
        :param folder_path: The path to the folder containing the data.
        :return: The loaded data.
        """
        sentences = self.__load_folder(folder_path)
        clean_data, text_enum = self.__clean_data(sentences)
        n_gram = self.__get_ngrams(clean_data)
        return n_gram, text_enum

    def evaluate_model(self, train_ngram, test_ngram, train_enum, test_enum):
        """
        Train the model with the given training data.
        :param train_path: The path to the training data.
        :return: mat_train: The one-hot encoded training data.
        """

        # Flatten the data
        train_n_gram_corpus = self.__flatten_data(train_ngram)

        # One-hot encoding
        mat_train = self.__one_hot_encoding(train_n_gram_corpus, train_ngram)

        mat_test = self.__one_hot_encoding(train_n_gram_corpus, test_ngram)

        # Evaluate the similarity between the two datasets
        max_similarity, average_similarity, comparison = self.__cosine_similarity(
            mat_test, test_enum, mat_train, train_enum
        )
        return max_similarity, average_similarity, comparison

    def AUC(self, average_similarity, text_results_catalog):
        """
        Calculate the Area Under the Curve (AUC) for the given threshold dictionary.
        :param text_results_catalog: The threshold dictionary.
        :return: The AUC value.
        """
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for key, value in average_similarity.items():
            if text_results_catalog[str(key)] == 1:
                if value > self.__similarity_limit:
                    TP += 1
                else:
                    FN += 1
            else:
                if value > self.__similarity_limit:
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

        AUC = (1 + TPR - FPR) / 2

        return (TP, FP, TN, FN, AUC)

    def print_max_similarity(self, max_similarity):
        """
        Print the maximum similarity between the test and training data.
        :return: None
        """
        current_key = -1
        print("Max similarity:")
        for key, value in max_similarity.items():
            if key[0] != current_key:
                current_key = key[0]
                print(f"-----Test file #{current_key}-----")
            print(
                f"Sentence #{int(key[1])+1} --> Train file #{int(value[0][0])+1}. Sentence #{int(value[0][1])+1}: {round((value[1])*100, 3)}%"
            )

    def print_average_similarity(self, average_similarity):
        """
        Print the average similarity between the test and training data.
        :return: None
        """
        print("Average similarity:")
        for key, value in average_similarity.items():
            flag = False
            if value > self.__similarity_limit:
                flag = True
            print(
                f"Test file #{int(key)+1} | Plagiarized: {flag} | Average similarity: {round((value)*100, 3)}%"
            )

    def print_comparison(self, comparison):
        """
        Print the comparison between the test and training data.
        :return: None
        """
        current_key = -1
        print("Comparison:")
        for key, value in comparison.items():
            if value[0][1] != current_key:
                current_key = value[0][1]
                print(f"-----Test file #{current_key}-----")
            print(
                f"Sentence #{int(value[1][1])+1} --> Train file #{int(value[1][0])+1}. Sentence #{int(value[1][1])+1}: {round((value[2])*100, 3)}%"
            )

    def __cosine_similarity(self, mat_test, test_enum, mat_train, train_enum):
        """
        Compare the test and training data using cosine similarity.
        :param mat_test: The one-hot encoded test data.
        :param test_enum: The enumerated test data.
        :return: The maximum similarity, average similarity, and comparison between the test and training data.
        """
        max_similarity = {}
        comparison = {}
        for i in range(len(mat_test)):
            for j in range(len(mat_train)):
                cosine_similarity = pairwise.cosine_similarity(
                    [mat_test[i]], [mat_train[j]]
                )
                comparison[(i, j)] = [
                    test_enum[i],
                    train_enum[j],
                    cosine_similarity[0][0],
                ]
                if tuple(test_enum[i]) not in max_similarity:
                    max_similarity[tuple(test_enum[i])] = [
                        train_enum[j],
                        cosine_similarity[0][0],
                    ]
                else:
                    if cosine_similarity[0][0] > max_similarity[tuple(test_enum[i])][1]:
                        max_similarity[tuple(test_enum[i])] = [
                            train_enum[j],
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

    def __load_folder(self, folder_path):
        """
        Load the data from the given folder path.
        :param folder_path: The path to the folder containing the data.
        :return: The loaded data.
        """
        data = []
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(folder_path, filename), "r", encoding="utf-8-sig"
                ) as f:
                    data.append(f.read())
        return data

    def __clean_data(self, data):
        """
        Clean the data from the given folder path.
        :param folder_path: The path to the folder containing the data.
        :return: The cleaned data.
        """
        preprocessor = Preprocessor()
        data, text_enum = preprocessor.clean_data(data)
        return data, text_enum

    def __get_ngrams(self, data):
        """
        Create n-grams from the given data.
        :param data: The data to create n-grams from.
        :return: The n-grams.
        """
        n_gram = []

        for text in data:
            n_gram.append(list(ngrams(text, self.__n_gram)))
        return n_gram

    def __flatten_data(self, data):
        """
        Flatten the given data.
        :param data: The data to flatten.
        :return: The flattened data.
        """
        return [word for text in data for word in text]

    def __one_hot_encoding(self, corpus, data):
        """
        Perform one-hot encoding on the given data.
        :param corpus: The corpus to encode.
        :param data: The data to encode.
        :return: The one-hot encoded data.
        """
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

    def __get_similarity_limit(self):
        """
        Get the similarity limit.
        :return: The similarity limit.
        """
        return self.__similarity_limit
    
    def __get_n_gram(self):
        """
        Get the n-gram.
        :return: The n-gram.
        """
        return self.__n_gram
