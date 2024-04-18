import re
import os
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from sklearn.metrics import pairwise


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
        self.evaluate(mat_train, mat_test, train_enum, test_enum, train_n_gram, test_n_gram)

        return []

    def evaluate(self, train_data, train_test, train_enum, test_enum, train_sentences, test_sentences):
        for i in range(len(train_data)):
            for j in range(len(train_test)):
                cosine_similarity = pairwise.cosine_similarity(
                    [train_test[j]], [train_data[i]]
                )
                if cosine_similarity[0][0] > 0.5:
                    print(f"Similarity detected:\nFID-{(test_enum[j][0])+1}.txt sentence {(test_enum[j][1]+1)} vs org-{(train_enum[i][0])+1}.txt sentence {(train_enum[i][1])+1}: {round((cosine_similarity[0][0])*100, 2)}%")
                    print(f"FID-{(test_enum[j][0])+1}.txt: {test_sentences[j]}")
                    print(f"org-{(train_enum[i][0])+1}.txt: {train_sentences[i]}")
                    print()

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


class Preprocessor:
    def __init__(self, data):
        self.__data = data
        self.__tokenized_words = []
        self.__text_enum = {}

    def clean_data(self):
        self.__text_enum = self.get_text_enum()
        self.__data = self.tokenize_data()
        self.__data = self.lower_case()
        self.__data = self.remove_non_word()
        self.__tokenized_words = self.tokenize_words()
        self.__tokenized_words = self.remove_stop_words()
        self.__tokenized_words = self.lemmatize_data()
        return self.__tokenized_words, self.__text_enum
    
    def get_text_enum(self):
        text_enum = {}
        global_count = 0
        for i, text in enumerate(self.__data):
            tokenized_text = sent_tokenize(text)
            count = 0
            for _ in tokenized_text:
                text_enum[global_count] = [i, count]
                count += 1
                global_count += 1

        return text_enum

    def tokenize_data(self):
        return [sent_tokenize(text) for text in self.__data]

    def lower_case(self):
        return [[sentence.lower() for sentence in text] for text in self.__data]

    def remove_non_word(self):
        return [
            [re.sub(r"[^\w\s]", "", sentence) for sentence in text]
            for text in self.__data
        ]

    def tokenize_words(self):
        return [word_tokenize(sentence) for text in self.__data for sentence in text]

    def remove_stop_words(self):
        stop_words = set(stopwords.words("english"))
        return [
            [word for word in text if word not in stop_words]
            for text in self.__tokenized_words
        ]

    def lemmatize_data(self):
        lemmatizer = WordNetLemmatizer()
        return [
            [lemmatizer.lemmatize(word) for word in text]
            for text in self.__tokenized_words
        ]


if __name__ == "__main__":
    train_path = "./train"
    test_path = "./test_dummy"
    n = 2
    nlp = NLPModel()
    nlp.run(train_path, test_path, n)
