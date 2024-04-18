from NLPModel import NLPModel
from Preprocessor import Preprocessor


train_path = "./train"
test_path = "./test"
n = 2
nlp = NLPModel()
nlp.run(train_path, test_path, n)
