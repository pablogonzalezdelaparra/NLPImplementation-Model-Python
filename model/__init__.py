import os
from nltk.util import ngrams
from sklearn.metrics import pairwise
from model.Preprocessor import Preprocessor
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

class NLPModel:
    """
    The NLPModel class is used to train and evaluate the model.
    """

    def __init__(self, similarity_limit=0.5, n_gram=2):
        """
        Initialize the NLPModel class.
        :param similarity_limit: The similarity limit.
        :param n_gram: The n-gram.
        """
        self.__similarity_limit = similarity_limit
        self.__n_gram = n_gram

    def initialize_bert_classifiers(self):
        self.__tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",)
        self.__model = AutoModel.from_pretrained("bert-base-uncased",output_hidden_states=True)

    def initialize_classifiers(self, train_data, num_groups=11, len_groups=10):
        subjects_groups = [[i] * len_groups for i in range(num_groups)]
        subjects = [subject for group in subjects_groups for subject in group]

        self.__pipeline = Pipeline(
            [("tfidf", TfidfVectorizer()), ("clf", LogisticRegression())]
        )
        self.__pipeline.fit(train_data, subjects)

    def classify_text(self, test_data):
        return self.__pipeline.predict(test_data)
    
    def find_similar_text(self, train_texts, test_texts, classifiers):
        def get_embeddings(text,token_length):
            tokens=self.__tokenizer(text,max_length=token_length,padding='max_length',truncation=True)
            output=self.__model(torch.tensor(tokens.input_ids).unsqueeze(0),
                        attention_mask=torch.tensor(tokens.attention_mask).unsqueeze(0)).hidden_states[-1]
            return torch.mean(output,axis=1).detach().numpy()


        def calculate_similarity(text1,text2,token_length=20):
            out1=get_embeddings(text1,token_length=token_length)
            out2=get_embeddings(text2,token_length=token_length)
            sim1= cosine_similarity(out1,out2)[0][0]
            return sim1

        max_similarity = {}
        for test_index, i in enumerate(classifiers):
            start_range = i * 10
            end_range = start_range + 10
            train_texts_temp = train_texts[start_range:end_range]
            for train_index, j in enumerate(train_texts_temp):
                test_texts[test_index]
                similarity = calculate_similarity(j,test_texts[test_index])
                if test_index not in max_similarity:
                    max_similarity[test_index] = [start_range + train_index, similarity]
                else:
                    if similarity > max_similarity[test_index][1]:
                        max_similarity[test_index] = [start_range + train_index, similarity]

        return max_similarity

    def prepare_text(self, folder_path):
        texts = self.__load_folder(folder_path)
        text_enum = self.__clean_data_texts(texts)
        return texts, text_enum

    def evaluate_model_text(self, train_ngram, test_ngram, train_enum, test_enum, max_similarity_text, train_data_enum, test_data_enum):   
        # Flatten the data
        train_n_gram_corpus = self.__flatten_data(train_ngram)

        # One-hot encoding
        mat_train = self.__one_hot_encoding(train_n_gram_corpus, train_ngram)

        mat_test = self.__one_hot_encoding(train_n_gram_corpus, test_ngram)

        # Evaluate the similarity between the two datasets
        max_similarity, average_similarity, comparison = self.__cosine_similarity_text(
            mat_train, mat_test, train_enum, test_enum, max_similarity_text, train_data_enum, test_data_enum
        )
        return max_similarity, average_similarity, comparison
    
    def __cosine_similarity_text(self, mat_train, mat_test, train_enum, test_enum, max_similarity_text, train_data_enum, test_data_enum):
        max_similarity = {}
        comparison = {}

        print(max_similarity_text)
        print(test_data_enum)
        print(test_enum)

        for i in range(len(mat_test)):
            num_test_text = test_data_enum[i][0]
            start_range = train_enum[max_similarity_text[num_test_text][0]][0]
            end_range = train_enum[max_similarity_text[num_test_text][0]][1]

            for j in range(start_range, end_range):
                cosine_similarity = pairwise.cosine_similarity(
                    [mat_test[i]], [mat_train[j]]
                )
                comparison[(i, j)] = [
                    test_data_enum[i],
                    train_data_enum[j],
                    cosine_similarity[0][0],
                ]
                if tuple(test_data_enum[i]) not in max_similarity:
                    max_similarity[tuple(test_data_enum[i])] = [
                        train_data_enum[j],
                        cosine_similarity[0][0],
                    ]
                else:
                    if cosine_similarity[0][0] > max_similarity[tuple(test_data_enum[i])][1]:
                        max_similarity[tuple(test_data_enum[i])] = [
                            train_data_enum[j],
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

    def prepare_sentences(self, folder_path):
        """
        Prepare the text data for the model.
        :param folder_path: The path to the folder containing the data.
        :return: n_gram: The n-grams.
        :return: text_enum: The enumerated text data.
        """
        texts = self.__load_folder(folder_path)
        sentences, text_enum = self.__clean_data(texts)
        n_gram = self.__get_ngrams(sentences)
        return n_gram, text_enum

    def evaluate_model(self, train_ngram, test_ngram, train_enum, test_enum):
        """
        Evaluate the model using the given data.
        :param train_ngram: The training n-grams.
        :param test_ngram: The test n-grams.
        :param train_enum: The enumerated training data.
        :param test_enum: The enumerated test data.
        :return: max_similarity: The maximum similarity between the test and training data.
        :return: average_similarity: The average similarity between the test and training data.
        :return: comparison: The comparison between the test and training data.
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
        Calculate the AUC.
        :param average_similarity: The average similarity between the test and training data.
        :param text_results_catalog: The catalog of the test data.
        :return: The AUC.
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
        :param max_similarity: The maximum similarity between the test and training data.
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
        :param average_similarity: The average similarity between the test and training data.
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
        :param comparison: The comparison between the test and training data.
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
        Calculate the cosine similarity between the test and training data.
        :param mat_test: The test data.
        :param test_enum: The enumerated test data.
        :param mat_train: The training data.
        :param train_enum: The enumerated training data.
        :return: max_similarity: The maximum similarity between the test and training data.
        :return: average_similarity: The average similarity between the test and training data.
        :return: comparison: The comparison between the test and training data.
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
        :return: The data.
        """
        data = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(folder_path, filename), "r", encoding="utf-8-sig"
                ) as f:
                    data.append(f.read())
        return data

    def __clean_data(self, data):
        """
        Clean the given data.
        :param data: The data to clean.
        :return: The cleaned data.
        """
        preprocessor = Preprocessor()
        data, text_enum = preprocessor.clean_data(data)
        return data, text_enum
    
    def __clean_data_texts(self, data):
        preprocessor = Preprocessor()
        text_enum = preprocessor.clean_data_text(data)
        return text_enum

    def __get_ngrams(self, data):
        """
        Get the n-grams of the given data.
        :param data: The data to get the n-grams of.
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
        :param corpus: The corpus of the data.
        :param data: The data to encode.
        :return: The encoded data.
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

    def get_similarity_limit(self):
        """
        Get the similarity limit.
        :return: The similarity limit.
        """
        return self.__similarity_limit

    def get_n_gram(self):
        """
        Get the n-gram.
        :return: The n-gram.
        """
        return self.__n_gram
