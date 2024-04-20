from unittest import TestCase
from model import NLPModel


class TestNLPModel(TestCase):

    def setUp(self):
        self.nlp = NLPModel()
        self.train_path = "./tests/train_data"
        self.test_path = "./tests/test_data"

    def test___load_folder_train(self):
        self.assertEqual(
            [
                "This is the first sentence of the second text. Now, this is another sentence but also of the second text.\n"
            ],
            self.nlp._NLPModel__load_folder(self.train_path),
        )

    def test___clean_data_train(self):
        self.assertEqual(
            (
                [
                    ["first", "sentence", "second", "text"],
                    ["another", "sentence", "also", "second", "text"],
                ],
                {0: [0, 0], 1: [0, 1]},
            ),
            self.nlp._NLPModel__clean_data(
                [
                    "This is the first sentence of the second text. Now, this is another sentence but also of the second text.\n"
                ],
            ),
        )

    def test___get_ngrams_train(self):
        self.assertEqual(
            [
                [
                    ("first", "sentence"),
                    ("sentence", "second"),
                    ("second", "text"),
                ],
                [
                    ("another", "sentence"),
                    ("sentence", "also"),
                    ("also", "second"),
                    ("second", "text"),
                ],
            ],
            self.nlp._NLPModel__get_ngrams(
                [
                    ["first", "sentence", "second", "text"],
                    ["another", "sentence", "also", "second", "text"],
                ],
            ),
        )

    def test___load_folder_test(self):
        self.assertEqual(
            [
                "This is the first sentence of the first text. Now, this is another sentence but also of the first text.\n"
            ],
            self.nlp._NLPModel__load_folder(self.test_path),
        )

    def test___clean_data_test(self):
        self.assertEqual(
            (
                [
                    ["first", "sentence", "first", "text"],
                    ["another", "sentence", "also", "first", "text"],
                ],
                {0: [0, 0], 1: [0, 1]},
            ),
            self.nlp._NLPModel__clean_data(
                [
                    "This is the first sentence of the first text. Now, this is another sentence but also of the first text.\n"
                ],
            ),
        )

    def test___get_ngrams_test(self):
        self.assertEqual(
            [
                [
                    ("first", "sentence"),
                    ("sentence", "first"),
                    ("first", "text"),
                ],
                [
                    ("another", "sentence"),
                    ("sentence", "also"),
                    ("also", "first"),
                    ("first", "text"),
                ],
            ],
            self.nlp._NLPModel__get_ngrams(
                [
                    ["first", "sentence", "first", "text"],
                    ["another", "sentence", "also", "first", "text"],
                ],
            ),
        )

    def test___flatten_data(self):
        self.assertEqual(
            [
                ("first", "sentence"),
                ("sentence", "second"),
                ("second", "text"),
                ("another", "sentence"),
                ("sentence", "also"),
                ("also", "second"),
                ("second", "text"),
            ],
            self.nlp._NLPModel__flatten_data(
                [
                    [
                        ("first", "sentence"),
                        ("sentence", "second"),
                        ("second", "text"),
                    ],
                    [
                        ("another", "sentence"),
                        ("sentence", "also"),
                        ("also", "second"),
                        ("second", "text"),
                    ],
                ],
            ),
        )

    def test___one_hot_encoding_train(self):
        self.assertEqual(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
            ],
            self.nlp._NLPModel__one_hot_encoding(
                [
                    ("first", "sentence"),
                    ("sentence", "second"),
                    ("second", "text"),
                    ("another", "sentence"),
                    ("sentence", "also"),
                    ("also", "second"),
                    ("second", "text"),
                ],
                [
                    [
                        ("first", "sentence"),
                        ("sentence", "second"),
                        ("second", "text"),
                    ],
                    [
                        ("another", "sentence"),
                        ("sentence", "also"),
                        ("also", "second"),
                        ("second", "text"),
                    ],
                ],
            ),
        )

    def test___one_hot_encoding_test(self):
        self.assertEqual(
            [
                [1, 1, 1, 0, 0, 0, 1],
                [0, 0, 1, 1, 1, 1, 1],
            ],
            self.nlp._NLPModel__one_hot_encoding(
                [
                    ("first", "sentence"),
                    ("sentence", "first"),
                    ("first", "text"),
                    ("another", "sentence"),
                    ("sentence", "also"),
                    ("also", "first"),
                    ("first", "text"),
                ],
                [
                    [
                        ("first", "sentence"),
                        ("sentence", "first"),
                        ("first", "text"),
                    ],
                    [
                        ("another", "sentence"),
                        ("sentence", "also"),
                        ("also", "first"),
                        ("first", "text"),
                    ],
                ],
            ),
        )

    def test___cosine_similarity(self):
        self.assertEqual(
            (
                {(0, 0): [[0, 0], 0.5], (0, 1): [[0, 1], 0.6324555320336758]},
                {0: 0.5662277660168379},
                {
                    (0, 0): [[0, 0], [0, 0], 0.5],
                    (0, 1): [[0, 0], [0, 1], 0.0],
                    (1, 0): [[0, 1], [0, 0], 0.0],
                    (1, 1): [[0, 1], [0, 1], 0.6324555320336758],
                },
            ),
            self.nlp._NLPModel__cosine_similarity(
                [[1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0]],
                {0: [0, 0], 1: [0, 1]},
                [[1, 1, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 1, 1]],
                {0: [0, 0], 1: [0, 1]},
            ),
        )

    def test_AUC(self):
        self.assertEqual(
            (1, 0, 0, 0, 1.0),
            self.nlp.AUC(
                {0: 0.5662277660168379},
                {
                    "0": 1,
                    "1": 1,
                    "2": 1,
                    "3": 1,
                    "4": 1,
                    "5": 1,
                    "6": 1,
                    "7": 1,
                    "8": 1,
                    "9": 1,
                    "10": 0,
                    "11": 0,
                    "12": 0,
                    "13": 0,
                    "14": 0,
                    "15": 0,
                    "16": 0,
                    "17": 0,
                    "18": 0,
                    "19": 0,
                },
            ),
        )

    def test___load_folder_train_empty(self):
        self.assertRaises(
            FileNotFoundError,
            self.nlp._NLPModel__load_folder,
            "",
        )

    def test___clean_data_train(self):
        self.assertEqual(
            (
                [
                    
                ],
                {},
            ),
            self.nlp._NLPModel__clean_data(
                [
                   
                ],
            ),
        )

    def test___get_ngrams_train_empty(self):
        self.assertEqual(
            [
                
            ],
            self.nlp._NLPModel__get_ngrams(
                [
                    
                ],
            ),
        )

    def test___flatten_data_empty(self):
        self.assertEqual(
            [
                
            ],
            self.nlp._NLPModel__flatten_data(
                [
                   
                ],
            ),
        )

    def test___one_hot_encoding_train_empty(self):
        self.assertEqual(
            [
               
            ],
            self.nlp._NLPModel__one_hot_encoding(
                [
                   
                ],
                [

                ],
            ),
        )

    def test___one_hot_encoding_test_empty(self):
        self.assertEqual(
            [
              
            ],
            self.nlp._NLPModel__one_hot_encoding(
                [
                    
                ],
                [
                   
                ],
            ),
        )

    def test___cosine_similarity_empty(self):
        self.assertEqual(
            (
                {},
                {},
                {
                    
                },
            ),
            self.nlp._NLPModel__cosine_similarity(
                [],
                {},
                [],
                {},
            ),
        )

    def test_AUC_empty(self):
        self.assertEqual(
            (0, 0, 0, 0, 0.5),
            self.nlp.AUC(
                {},
                {
                   
                },
            ),
        )
