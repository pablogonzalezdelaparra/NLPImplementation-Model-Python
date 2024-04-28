import random
import os
from Preprocessor import Preprocessor

def load_folder(folder_path):
        data = []

        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith(".txt"):
                with open(
                    os.path.join(folder_path, filename), "r", encoding="utf-8-sig"
                ) as f:
                    data.append(f.read())
        return data

def unorder_data(folder):
    # Load the data
    original_data = load_folder(folder)
    preprocessor = Preprocessor()
    data, _ = preprocessor.categorize_sentences(original_data)

    unorder_data = []
    for sentences in data:
        # Copy the original list of sentences
        new_sentences = sentences[:]
        
        # Determine how many sentences to change
        if len(sentences) < 4:
            num_sentences_to_change = 2
        else:
            num_sentences_to_change = random.randint(2, len(sentences)//2)

        # Create a list of indices to shuffle
        indices_to_shuffle = random.sample(range(len(sentences)), num_sentences_to_change)

        # Keep track of used positions
        used_sentences = set()
        used_indices = set()

        for index in indices_to_shuffle:
            # Pick a random index to swap with
            new_index = random.randint(0, len(sentences)-1)
            while new_index in used_indices:
                new_index = random.randint(0, len(sentences)-1)
            used_indices.add(new_index)

            # Swap the sentences
            new_sentences[index], new_sentences[new_index] = new_sentences[new_index], new_sentences[index]

            # Keep track of which sentences have been changed
            used_sentences.add(index)
            used_sentences.add(new_index)
        unorder_data.append(new_sentences)

    final_data = []
    for data in unorder_data:
        # join the sentences back together
        final_data.append(" ".join(data))

    return final_data

def insert_replace_data(folder, insert_folder):
    # Load the data
    original_data = load_folder(folder)
    insert_sentences = load_folder(insert_folder)
    preprocessor = Preprocessor()
    data, _ = preprocessor.categorize_sentences(original_data)
    insert_data, _ = preprocessor.categorize_sentences(insert_sentences)

    modified_data = []
    for sentences in data:
        # Copy the original list of sentences
        new_sentences = sentences[:]
        
        # Determine how many sentences to change
        if len(sentences) < 4:
            num_sentences_to_change = 2
        else:
            num_sentences_to_change = random.randint(2, len(sentences)//2)

        # Create a list of indices to shuffle
        indices_to_shuffle = random.sample(range(len(sentences)), num_sentences_to_change)

        for index in indices_to_shuffle:
            random_insert = random.choice(insert_data)
            random_insert_sentence = random.choice(random_insert)

            random_number = random.random()
            if random_number < 0.5:
                new_sentences[index] = random_insert_sentence
            else:
                new_sentences.insert(index, random_insert_sentence)
        modified_data.append(new_sentences)

    final_data = []
    for data in modified_data:
        # join the sentences back together
        final_data.append(" ".join(data))

    return final_data

if __name__ == "__main__":

    # Create the generated_data folder
    if not os.path.exists("generated_data"):
            os.makedirs("generated_data")

    def unorder():
        unordered_data = unorder_data("./train_data")

        if not os.path.exists("generated_data/unordered_data"):
            os.makedirs("generated_data/unordered_data")

        for i, text in enumerate(unordered_data):
            with open(f"generated_data/unordered_data/FID-{i}-1.txt", "w", encoding="utf-8") as f:
                f.write(text)

    def insert_replace():
        inserted_replaced_data = insert_replace_data("./train_data", "./test_data_insert")

        # save the modified data in a new folder and a subfolder
        if not os.path.exists("generated_data/insert_replace_data"):
            os.makedirs("generated_data/insert_replace_data")

        for i, text in enumerate(inserted_replaced_data):
            with open(f"generated_data/insert_replace_data/FID-{i}-2.txt", "w", encoding="utf-8") as f:
                f.write(text)
        
    # insert_replace()
    # unorder()
