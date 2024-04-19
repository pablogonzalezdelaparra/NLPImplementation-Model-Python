from NLPModel import NLPModel

train_path = "./train"
test_path = "./test"
n_gram = 2

nlp = NLPModel()
nlp.train(train_path, n_gram)
nlp.evaluate(test_path)
nlp.print_average_similarity()
