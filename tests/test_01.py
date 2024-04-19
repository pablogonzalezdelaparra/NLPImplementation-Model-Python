from unittest import TestCase
from model import NLPModel

class TestNLPModel(TestCase):

    def setUp(self):
        self.nlp = NLPModel()
        self.train_path = "./train_data"
        self.test_path = "./test_data"

    def test_0(self):
        self.assertEqual(
            [],
            self.nlp.train(self.train_path)
        )
