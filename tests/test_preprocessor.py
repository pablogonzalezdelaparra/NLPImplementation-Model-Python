from unittest import TestCase
from model import NLPModel
from model.Preprocessor import Preprocessor


class TestPreprocessor(TestCase):

    def setUp(self):
        self.nlp = NLPModel()
        self.p = Preprocessor(
            [
                "Artificial intelligence, often abbreviated as AI, revolutionizes how we interact with technology. From powering virtual assistants to driving autonomous vehicles, AI systems mimic human intelligence to perform tasks efficiently.",
                "Its applications span across industries, from healthcare to finance, enhancing productivity and decision-making processes. As AI continues to evolve, its potential to reshape society's dynamics and advance human progress remains unparalleled.",
            ]
        )
        self.train_path = "./train_data"
        self.test_path = "./test_data"

    def test___get_text_enum(self):
        self.assertEqual(
            {0: [0, 0], 1: [0, 1], 2: [1, 0], 3: [1, 1]},
            self.p._Preprocessor__get_text_enum(),
        )

    def test___tokenize_data(self):
        self.assertEqual(
            [
                [
                    "Artificial intelligence, often abbreviated as AI, revolutionizes how we interact with technology.",
                    "From powering virtual assistants to driving autonomous vehicles, AI systems mimic human intelligence to perform tasks efficiently.",
                ],
                [
                    "Its applications span across industries, from healthcare to finance, enhancing productivity and decision-making processes.",
                    "As AI continues to evolve, its potential to reshape society's dynamics and advance human progress remains unparalleled.",
                ],
            ],
            self.p._Preprocessor__tokenize_data(),
        )

    # def test___lower_case(self):
    #     ...
