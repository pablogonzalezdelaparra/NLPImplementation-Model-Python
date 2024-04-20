from unittest import TestCase
from model import NLPModel
from model.Preprocessor import Preprocessor


class TestPreprocessor(TestCase):

    def setUp(self):
        self.p = Preprocessor()

    def test_clean_data(self):
        self.assertEqual(
            (
                [
                    ["first", "sentence", "first", "text"],
                    ["another", "sentence", "also", "first", "text"],
                    ["first", "sentence", "second", "text"],
                    ["another", "sentence", "also", "second", "text"],
                ],
                {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]},
            ),
            self.p.clean_data(
                [
                    "This is the first sentence of the first text. Now, this is another sentence but also of the first text.",
                    "This is the first sentence of the second text. Now, this is another sentence but also of the second text.",
                ]
            ),
        )

    def test___get_text_enum(self):
        self.assertEqual(
            {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]},
            self.p._Preprocessor__get_text_enum(
                [
                    "This is the first sentence of the first text. Now, this is another sentence but also of the first text.",
                    "This is the first sentence of the second text. Now, this is another sentence but also of the second text.",
                ]
            ),
        )

    def test___tokenize_data(self):
        self.assertEqual(
            [
                [
                    "This is the first sentence of the first text.",
                    "Now, this is another sentence but also of the first text.",
                ],
                [
                    "This is the first sentence of the second text.",
                    "Now, this is another sentence but also of the second text.",
                ],
            ],
            self.p._Preprocessor__tokenize_data(
                [
                    "This is the first sentence of the first text. Now, this is another sentence but also of the first text.",
                    "This is the first sentence of the second text. Now, this is another sentence but also of the second text.",
                ]
            ),
        )

    def test___lower_case(self):
        self.assertEqual(
            [
                [
                    "this is the first sentence of the first text.",
                    "now, this is another sentence but also of the first text.",
                ],
                [
                    "this is the first sentence of the second text.",
                    "now, this is another sentence but also of the second text.",
                ],
            ],
            self.p._Preprocessor__lower_case(
                [
                    [
                        "This is the first sentence of the first text.",
                        "Now, this is another sentence but also of the first text.",
                    ],
                    [
                        "This is the first sentence of the second text.",
                        "Now, this is another sentence but also of the second text.",
                    ],
                ]
            ),
        )

    def test___remove_non_word(self):
        self.assertEqual(
            [
                [
                    "This is the first sentence of the first text",
                    "Now this is another sentence but also of the first text",
                ],
                [
                    "This is the first sentence of the second text",
                    "Now this is another sentence but also of the second text",
                ],
            ],
            self.p._Preprocessor__remove_non_word(
                [
                    [
                        "This is the first sentence of the first text.",
                        "Now, this is another sentence but also of the first text.",
                    ],
                    [
                        "This is the first sentence of the second text.",
                        "Now, this is another sentence but also of the second text.",
                    ],
                ]
            ),
        )

    def test___tokenize_words(self):
        self.assertEqual(
            [
                [
                    "This",
                    "is",
                    "the",
                    "first",
                    "sentence",
                    "of",
                    "the",
                    "first",
                    "text",
                    ".",
                ],
                [
                    "Now",
                    ",",
                    "this",
                    "is",
                    "another",
                    "sentence",
                    "but",
                    "also",
                    "of",
                    "the",
                    "first",
                    "text",
                    ".",
                ],
                [
                    "This",
                    "is",
                    "the",
                    "first",
                    "sentence",
                    "of",
                    "the",
                    "second",
                    "text",
                    ".",
                ],
                [
                    "Now",
                    ",",
                    "this",
                    "is",
                    "another",
                    "sentence",
                    "but",
                    "also",
                    "of",
                    "the",
                    "second",
                    "text",
                    ".",
                ],
            ],
            self.p._Preprocessor__tokenize_words(
                [
                    [
                        "This is the first sentence of the first text.",
                        "Now, this is another sentence but also of the first text.",
                    ],
                    [
                        "This is the first sentence of the second text.",
                        "Now, this is another sentence but also of the second text.",
                    ],
                ]
            ),
        )

    def test___remove_stop_words(self):
        self.assertEqual(
            [
                ["This", "first", "sentence", "first", "text", "."],
                ["Now", ",", "another", "sentence", "also", "first", "text", "."],
                ["This", "first", "sentence", "second", "text", "."],
                ["Now", ",", "another", "sentence", "also", "second", "text", "."],
            ],
            self.p._Preprocessor__remove_stop_words(
                [
                    [
                        "This",
                        "is",
                        "the",
                        "first",
                        "sentence",
                        "of",
                        "the",
                        "first",
                        "text",
                        ".",
                    ],
                    [
                        "Now",
                        ",",
                        "this",
                        "is",
                        "another",
                        "sentence",
                        "but",
                        "also",
                        "of",
                        "the",
                        "first",
                        "text",
                        ".",
                    ],
                    [
                        "This",
                        "is",
                        "the",
                        "first",
                        "sentence",
                        "of",
                        "the",
                        "second",
                        "text",
                        ".",
                    ],
                    [
                        "Now",
                        ",",
                        "this",
                        "is",
                        "another",
                        "sentence",
                        "but",
                        "also",
                        "of",
                        "the",
                        "second",
                        "text",
                        ".",
                    ],
                ],
            ),
        )

    def test___lemmatize_data(self):
        self.assertEqual(
            [
                [
                    "This",
                    "is",
                    "the",
                    "first",
                    "sentence",
                    "of",
                    "the",
                    "first",
                    "text",
                    ".",
                ],
                [
                    "Now",
                    ",",
                    "this",
                    "is",
                    "another",
                    "sentence",
                    "but",
                    "also",
                    "of",
                    "the",
                    "first",
                    "text",
                    ".",
                ],
                [
                    "This",
                    "is",
                    "the",
                    "first",
                    "sentence",
                    "of",
                    "the",
                    "second",
                    "text",
                    ".",
                ],
                [
                    "Now",
                    ",",
                    "this",
                    "is",
                    "another",
                    "sentence",
                    "but",
                    "also",
                    "of",
                    "the",
                    "second",
                    "text",
                    ".",
                ],
            ],
            self.p._Preprocessor__lemmatize_data(
                [
                    [
                        "This",
                        "is",
                        "the",
                        "first",
                        "sentence",
                        "of",
                        "the",
                        "first",
                        "text",
                        ".",
                    ],
                    [
                        "Now",
                        ",",
                        "this",
                        "is",
                        "another",
                        "sentence",
                        "but",
                        "also",
                        "of",
                        "the",
                        "first",
                        "text",
                        ".",
                    ],
                    [
                        "This",
                        "is",
                        "the",
                        "first",
                        "sentence",
                        "of",
                        "the",
                        "second",
                        "text",
                        ".",
                    ],
                    [
                        "Now",
                        ",",
                        "this",
                        "is",
                        "another",
                        "sentence",
                        "but",
                        "also",
                        "of",
                        "the",
                        "second",
                        "text",
                        ".",
                    ],
                ]
            ),
        )

    def test_clean_data_empty(self):
        self.assertEqual(
            (
                [],
                {},
            ),
            self.p.clean_data([]),
        )

    def test___get_text_enum_empty(self):
        self.assertEqual(
            {},
            self.p._Preprocessor__get_text_enum([]),
        )

    def test___tokenize_data_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__tokenize_data([]),
        )

    def test___lower_case_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__lower_case([]),
        )

    def test___remove_non_word_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__remove_non_word([]),
        )

    def test___tokenize_words_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__tokenize_words([]),
        )

    def test___remove_stop_words_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__remove_stop_words(
                [],
            ),
        )

    def test___lemmatize_data_empty(self):
        self.assertEqual(
            [],
            self.p._Preprocessor__lemmatize_data([]),
        )
