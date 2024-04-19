from model import NLPModel

train_path = "./train_data"
test_path = "./test_data"

nlp = NLPModel()
nlp.train(train_path)
nlp.evaluate(test_path)
nlp.print_average_similarity()
