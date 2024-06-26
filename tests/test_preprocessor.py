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

    def test_clean_data_all_caps(self):
        self.assertEqual(
            (
                [
                    ["first", "short", "sentence"],
                    ["second", "short", "sentence"],
                    ["third", "short", "sentence", "capital", "letter"],
                    ["fourth", "sentence"],
                ],
                {0: [0, 0], 1: [1, 0], 2: [2, 0], 3: [3, 0]},
            ),
            self.p.clean_data(
                [
                    "A FIRST SHORT SENTENCE",
                    "THIS IS A SECOND SHORT SENTENCE",
                    "A THIRD SHORT SENTENCE, ALL IN CAPITAL LETTERS",
                    "FOURTH SENTENCE",
                ]
            ),
        )

    def test___get_text_enum_all_caps(self):
        self.assertEqual(
            {0: [0, 0], 1: [1, 0], 2: [2, 0], 3: [3, 0]},
            self.p._Preprocessor__get_text_enum(
                [
                    "A FIRST SHORT SENTENCE",
                    "THIS IS A SECOND SHORT SENTENCE",
                    "A THIRD SHORT SENTENCE, ALL IN CAPITAL LETTERS",
                    "FOURTH SENTENCE",
                ]
            ),
        )

    def test___tokenize_data_all_caps(self):
        self.assertEqual(
            [
                ["A FIRST SHORT SENTENCE"],
                ["THIS IS A SECOND SHORT SENTENCE"],
                ["A THIRD SHORT SENTENCE, ALL IN CAPITAL LETTERS"],
                ["FOURTH SENTENCE"],
            ],
            self.p._Preprocessor__tokenize_data(
                [
                    "A FIRST SHORT SENTENCE",
                    "THIS IS A SECOND SHORT SENTENCE",
                    "A THIRD SHORT SENTENCE, ALL IN CAPITAL LETTERS",
                    "FOURTH SENTENCE",
                ]
            ),
        )

    def test___lower_case_all_caps(self):
        self.assertEqual(
            [
                ["a first short sentence"],
                ["this is a second short sentence"],
                ["a third short sentence, all in capital letters"],
                ["fourth sentence"],
            ],
            self.p._Preprocessor__lower_case(
                [
                    ["A FIRST SHORT SENTENCE"],
                    ["THIS IS A SECOND SHORT SENTENCE"],
                    ["A THIRD SHORT SENTENCE, ALL IN CAPITAL LETTERS"],
                    ["FOURTH SENTENCE"],
                ],
            ),
        )

    def test_clean_data_symbols(self):
        self.assertEqual(
            (
                [["sentence", "many", "symbol"], ["second", "sentence", "symbol"]],
                {0: [0, 0], 1: [0, 1]},
            ),
            self.p.clean_data(
                [
                    "a & sentence # with, many () symbols ? ¡ / $"
                    "$$$$ second //// sentence & with = symbols"
                ]
            ),
        )

    def test___get_text_enum_symbols(self):
        self.assertEqual(
            {0: [0, 0], 1: [0, 1]},
            self.p._Preprocessor__get_text_enum(
                [
                    "a & sentence # with, many () symbols ? ¡ / $"
                    "$$$$ second //// sentence & with = symbols"
                ]
            ),
        )

    def test___tokenize_data_symbols(self):
        self.assertEqual(
            [
                [
                    "a & sentence # with, many () symbols ?",
                    "¡ / $$$$$ second //// sentence & with = symbols",
                ]
            ],
            self.p._Preprocessor__tokenize_data(
                [
                    "a & sentence # with, many () symbols ? ¡ / $"
                    "$$$$ second //// sentence & with = symbols"
                ]
            ),
        )

    def test___lower_case_symbols(self):
        self.assertEqual(
            [
                [
                    "a & sentence # with, many () symbols ?",
                    "¡ / $$$$$ second //// sentence & with = symbols",
                ]
            ],
            self.p._Preprocessor__lower_case(
                [
                    [
                        "a & sentence # with, many () symbols ?",
                        "¡ / $$$$$ second //// sentence & with = symbols",
                    ]
                ]
            ),
        )

    def test___remove_non_word_symbols(self):
        self.assertEqual(
            [
                [
                    "a  sentence  with many  symbols ",
                    "   second  sentence  with  symbols",
                ]
            ],
            self.p._Preprocessor__remove_non_word(
                [
                    [
                        "a & sentence # with, many () symbols ?",
                        "¡ / $$$$$ second //// sentence & with = symbols",
                    ]
                ],
            ),
        )

    def test___tokenize_words_symbols(self):
        self.assertEqual(
            [
                ["a", "sentence", "with", "many", "symbols"],
                ["second", "sentence", "with", "symbols"],
            ],
            self.p._Preprocessor__tokenize_words(
                [
                    [
                        "a  sentence  with many  symbols ",
                        "   second  sentence  with  symbols",
                    ]
                ]
            ),
        )

    def test_clean_data_trailing_spaces(self):
        self.assertEqual(
            (
                [["lot", "trailing", "space"], ["normal", "sentence", "space"]],
                {0: [0, 0], 1: [1, 0]},
            ),
            self.p.clean_data(
                [
                    "          a lot of trailing       spaces                       here    .",
                    "                             a normal sentence                   with spaces              . ",
                ]
            ),
        )

    def test___get_text_enum_trailing_spaces(self):
        self.assertEqual(
            {0: [0, 0], 1: [1, 0]},
            self.p._Preprocessor__get_text_enum(
                [
                    "          a lot of trailing       spaces                       here    .",
                    "                             a normal sentence                   with spaces              . ",
                ]
            ),
        )

    def test___tokenize_data_trailing_spaces(self):
        self.assertEqual(
            [
                [
                    "          a lot of trailing       spaces                       here    ."
                ],
                [
                    "                             a normal sentence                   with spaces              ."
                ],
            ],
            self.p._Preprocessor__tokenize_data(
                [
                    "          a lot of trailing       spaces                       here    .",
                    "                             a normal sentence                   with spaces              . ",
                ]
            ),
        )

    def test___lower_case_trailing_spaces(self):
        self.assertEqual(
            [
                [
                    "          a lot of trailing       spaces                       here    ."
                ],
                [
                    "                             a normal sentence                   with spaces              ."
                ],
            ],
            self.p._Preprocessor__lower_case(
                [
                    [
                        "          a lot of trailing       spaces                       here    ."
                    ],
                    [
                        "                             a normal sentence                   with spaces              ."
                    ],
                ],
            ),
        )

    def test___remove_non_word_trailing_spaces(self):
        self.assertEqual(
            [
                [
                    "          a lot of trailing       spaces                       here    "
                ],
                [
                    "                             a normal sentence                   with spaces              "
                ],
            ],
            self.p._Preprocessor__remove_non_word(
                [
                    [
                        "          a lot of trailing       spaces                       here    ."
                    ],
                    [
                        "                             a normal sentence                   with spaces              ."
                    ],
                ],
            ),
        )

    def test___tokenize_words_trailing_spaces(self):
        self.assertEqual(
            [
                ["a", "lot", "of", "trailing", "spaces", "here"],
                ["a", "normal", "sentence", "with", "spaces"],
            ],
            self.p._Preprocessor__tokenize_words(
                [
                    [
                        "          a lot of trailing       spaces                       here    "
                    ],
                    [
                        "                             a normal sentence                   with spaces              "
                    ],
                ],
            ),
        )
