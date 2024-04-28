import os
import csv
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def load_data_from_folder(folder_path, label):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                sentences = sent_tokenize(text)  # Tokenize the text into sentences
                for sentence in sentences:
                    data.append((sentence, label))
    return data

def create_csv(folder_paths, labels, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['text', 'label'])
        
        for folder_path, label in zip(folder_paths, labels):
            data = load_data_from_folder(folder_path, label)
            writer.writerows(data)

if __name__ == "__main__":
    folder_paths = ['./generated_data/insert_replace_data', './generated_data/paraphrase_data', './generated_data/unordered_data']
    labels = [0, 1, 2]
    output_file = "data_s.csv"
    
    create_csv(folder_paths, labels, output_file)
    print("CSV file created successfully.")
