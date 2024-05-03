import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class Preprocessor:
    """
    The Preprocessor class is used to clean the data.
    """

    def __init__(self):
        """
        Initialize the Preprocessor class.
        """
        pass

    def clean_data(self, data):
        """
        Clean the data.
        :param data: The data to clean.
        :return: The cleaned data.
        :return: The enumeration of the text.
        """
        text_enum = self.__get_text_enum(data)
        data = self.__tokenize_data(data)
        data = self.__lower_case(data)
        data = self.__remove_non_word(data)
        data = self.__tokenize_words(data)
        data = self.__remove_stop_words(data)
        data = self.__lemmatize_data(data)
        return data, text_enum

    def categorize_sentences(self, data):
        text_enum = self.__get_text_enum(data)
        data = self.__tokenize_data(data)
        return data, text_enum
    
    def clean_data_text(self, data):
        text_enum = self.__get_text_enum_text(data)
        return text_enum

    def __get_text_enum(self, data):
        """
        Get the enumeration of the text.
        :param data: The data to enumerate.
        :return: The enumeration of the text.
        """
        text_enum = {}
        global_count = 0
        for i, text in enumerate(data):
            tokenized_text = sent_tokenize(text)
            count = 0
            for _ in tokenized_text:
                text_enum[global_count] = [i, count]
                count += 1
                global_count += 1

        return text_enum
    
    def __get_text_enum_text(self, data):
        """
        Get the enumeration of the text.
        :param data: The data to enumerate.
        :return: The enumeration of the text.
        """
        text_enum = {}
        global_count = 0
        for i, text in enumerate(data):
            tokenized_text = sent_tokenize(text)
            count = len(tokenized_text)
            text_enum[i] = [global_count, global_count + count]
            global_count += count

        return text_enum

    def __tokenize_data(self, data):
        """
        Tokenize the data.
        :param data: The data to tokenize.
        :return: The tokenized data.
        """
        return [sent_tokenize(text) for text in data]

    def __lower_case(self, data):
        """
        Convert the data to lower case.
        :param data: The data to convert to lower case.
        :return: The data in lower case.
        """
        return [[sentence.lower() for sentence in text] for text in data]

    def __remove_non_word(self, data):
        """
        Remove non-word characters from the data.
        :param data: The data to remove non-word characters from.
        :return: The data with non-word characters removed.
        """
        return [
            [
                re.sub(
                    r"[-]|([^\w\s])",
                    lambda x: " " if x.group(1) == "-" else "",
                    sentence,
                )
                for sentence in text
            ]
            for text in data
        ]

    def __tokenize_words(self, data):
        """
        Tokenize the words in the data.
        :param data: The data to tokenize.
        :return: The tokenized data.
        """
        return [word_tokenize(sentence) for text in data for sentence in text]

    def __remove_stop_words(self, data):
        """
        Remove stop words from the data.
        :param data: The data to remove stop words from.
        :return: The data with stop words removed.
        """
        stop_words = set(stopwords.words("english"))
        return [
            [word for word in text if word not in stop_words]
            for text in data
        ]

    def __lemmatize_data(self, data):
        """
        Lemmatize the data.
        :param data: The data to lemmatize.
        :return: The lemmatized data.
        """
        lemmatizer = WordNetLemmatizer()
        return [
            [lemmatizer.lemmatize(word) for word in text]
            for text in data
        ]
