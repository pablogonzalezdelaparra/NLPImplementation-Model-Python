from model import NLPModel

train_path = "./train_data"
test_path = "./test_data"
threshold_dic = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 1, "5": 1, "6": 1, "7": 1, "8": 1, "9": 1, "10": 0, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0}

nlp = NLPModel()
nlp.train(train_path)
nlp.evaluate(test_path)
nlp.print_average_similarity()
nlp.AUC(threshold_dic)
