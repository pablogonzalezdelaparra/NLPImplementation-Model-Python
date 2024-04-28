from model import NLPModel

# Define the paths
train_path = "./train_data/"
test_path = "./generated_data/insert_replace_data"
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
train_data, train_enum = nlp.prepare_text(train_path)
test_data, test_enum = nlp.prepare_text(test_path)

# Evaluate the model
max_similarity, average_similarity, comparison = nlp.evaluate_model(
    train_data, test_data, train_enum, test_enum
)
nlp.AUC(average_similarity, text_results_catalog)

# Print the results
nlp.print_average_similarity(average_similarity)
# nlp.print_max_similarity(max_similarity)
# nlp.print_comparison(comparison)

"""
# Tag the data
insert_replace_data_enum, insert_replace_enum = nlp.categorize_text_sentences(insert_replace_path)
paraphrase_data_enum, paraphrase_enum = nlp.categorize_text_sentences(paraphrase_path)
unordered_data_enum, unordered_enum = nlp.categorize_text_sentences(unordered_path)
original_data_enum, original_enum = nlp.categorize_text_sentences(original_path)

# Compare lengths
print(len(insert_replace_data_enum))
print(len(paraphrase_data_enum))
print(len(unordered_data_enum))
print(len(original_data_enum))
"""
