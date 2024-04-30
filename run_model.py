from model import NLPModel

# Define the paths
train_path = "./train_data/"
test_path = "./test_data/"
text_results_catalog = {
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
}
nlp = NLPModel(0.5, 3)

# Prepare the data (NLP)
"""
train_data, train_enum = nlp.prepare_text(train_path)
test_data, test_enum = nlp.prepare_text(test_path)

# Evaluate the model (NLP)
max_similarity, average_similarity, comparison = nlp.evaluate_model(
    train_data, test_data, train_enum, test_enum
)
"""
# nlp.AUC(average_similarity, text_results_catalog)

# Prepare the data (BERT)
train_data_bert, train_enum_bert = nlp.categorize_text_sentences(train_path)
test_data_bert, test_enum_bert = nlp.categorize_text_sentences(test_path)

# Evaluate the model (BERT)
max_similarity_bert, average_similarity_bert, comparison_bert = nlp.evaluate_model_bert(
    train_data_bert, test_data_bert, train_enum_bert, test_enum_bert
)
# nlp.AUC(average_similarity_bert, text_results_catalog)

# Print the results
print(f"NLP\n")
# nlp.print_average_similarity(average_similarity)
print(f"BERT\n")
#nlp.print_average_similarity(average_similarity_bert)
nlp.print_max_similarity(max_similarity_bert)
# nlp.print_comparison(comparison)
