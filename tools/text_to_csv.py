
import os
import csv

def load_data_from_folder(folder_path, train_path, label):
    train_data = []
    data = []

    for filename in sorted(os.listdir(train_path)):
        if filename.endswith('.txt'):
            with open(os.path.join(train_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                train_data.append(text)

    for index, filename in enumerate(sorted(os.listdir(folder_path))):
        if filename.endswith('.txt'):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                text = file.read().strip()
                # train_index = (index // 2) % len(train_data)
                data.append((text, train_data[index], label))
    return data

def create_csv(folder_paths, train_path, labels, output_file):
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['plagiarized', 'original', 'is_plagiarized'])

        for folder_path, label in zip(folder_paths, labels):
            data = load_data_from_folder(folder_path, train_path, label)
            writer.writerows(data)

if __name__ == "__main__":
    folder_paths = ['./test_data']
    output_file = "data.csv"
    train_path = "./train_data"
    
    create_csv(folder_paths, train_path, labels, output_file)
    print("CSV file created successfully.")
