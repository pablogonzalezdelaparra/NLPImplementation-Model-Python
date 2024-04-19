import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    def __init__(self, data):
        self.__data = data
        self.__tokenized_words = []
        self.__text_enum = {}

    def clean_data(self):
        self.__text_enum = self.__get_text_enum()
        self.__data = self.__tokenize_data()
        self.__data = self.__lower_case()
        self.__data = self.__remove_non_word()
        self.__tokenized_words = self.__tokenize_words()
        self.__tokenized_words = self.__remove_stop_words()
        self.__tokenized_words = self.__lemmatize_data()
        return self.__tokenized_words, self.__text_enum

    def __get_text_enum(self):
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

    def __tokenize_data(self):
        return [sent_tokenize(text) for text in self.__data]

    def __lower_case(self):
        return [[sentence.lower() for sentence in text] for text in self.__data]

    def __remove_non_word(self):
        return [
            [re.sub(r"[^\w\s]", "", sentence) for sentence in text]
            for text in self.__data
        ]

    def __tokenize_words(self):
        return [word_tokenize(sentence) for text in self.__data for sentence in text]

    def __remove_stop_words(self):
        stop_words = set(stopwords.words("english"))
        return [
            [word for word in text if word not in stop_words]
            for text in self.__tokenized_words
        ]

    def __lemmatize_data(self):
        lemmatizer = WordNetLemmatizer()
        return [
            [lemmatizer.lemmatize(word) for word in text]
            for text in self.__tokenized_words
        ]
