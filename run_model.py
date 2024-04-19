from NLPModel import NLPModel

train_path = "./train"
test_path = "./test"

nlp = NLPModel()
nlp.train(train_path)
nlp.evaluate(test_path)
nlp.print_average_similarity()
