from model import NLPModel

# Define the paths
train_path = "./train_data"
test_path = "./test_data"
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

# Prepare the data
train_data, train_data_enum = nlp.prepare_sentences(train_path)
test_data, test_data_enum = nlp.prepare_sentences(test_path)

# Evaluate the model
max_similarity, average_similarity, comparison = nlp.evaluate_model(
    train_data, test_data, train_data_enum, test_data_enum
)
AUC = nlp.AUC(average_similarity, text_results_catalog)

# Print the results
print(f"\nNLP")
print(f"AUC: {AUC}")
nlp.print_average_similarity(average_similarity)
nlp.print_max_similarity(max_similarity)

# Prepare the text
train_texts, train_texts_enum = nlp.prepare_text(train_path)
test_texts, test_texts_enum = nlp.prepare_text(test_path)

# Initialize classifiers with categorized
nlp.initialize_classifiers(train_texts, 11, 10)

# Initialize BERT Model
nlp.initialize_bert_classifiers()

# Classify the test data
classifiers = nlp.classify_text(test_texts)

# Find the most similar text
max_similarity = nlp.find_similar_text(
    train_texts, test_texts, classifiers
)

# Evaluate the model
max_similarity_texts, average_similarity_texts, comparison_texts = nlp.evaluate_model_text(
    train_data, test_data, train_texts_enum, test_texts_enum, max_similarity, train_data_enum, test_data_enum
)
AUC_NLP2O = nlp.AUC(average_similarity_texts, text_results_catalog)

# Print the results
print(f"\nNLP 2.0")
print(f"AUC: {AUC_NLP2O}")
nlp.print_average_similarity(average_similarity_texts)
nlp.print_max_similarity(max_similarity_texts)
# nlp.print_comparison(comparison)
