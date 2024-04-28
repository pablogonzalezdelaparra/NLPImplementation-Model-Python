import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from model.Preprocessor import Preprocessor

# Define the dataset class
class PlagiarismDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Function to fine-tune the model
def fine_tune_model(model, train_loader, val_loader, optimizer, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_loader)
        
        model.eval()
        val_preds = []
        val_labels = []
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label']
            
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
            val_preds.extend(preds)
            val_labels.extend(labels)
        
        val_accuracy = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")


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

# Example data
original_texts = [
    "This is an original abstract.",
    "Another original abstract.",
    "Yet another original abstract.",
    "Wow, another original abstract.",
    "This is the last original abstract.",
    "This is a plagiarized abstract."
]  # List of original abstracts
plagiarized_texts = [
    "This is an original abstract.",
    "Another original abstract.",
    "Yet another original abstract.",
    "Wow, another original abstract.",
    "This is the last original abstract.",
    "This is a plagiarized abstract."
]  # List of plagiarized abstracts


original_data = load_folder("./generated_data/unordered_data/")
preprocessor = Preprocessor()
original_texts, _ = preprocessor.categorize_sentences(original_data)

original_texts = original_texts[:50]

print(len(original_texts))

original_data = load_folder("./generated_data/paraphrase_data/")
preprocessor = Preprocessor()
paraphrased_data, _ = preprocessor.categorize_sentences(original_data)

paraphrased_data = paraphrased_data[:60]

print(len(paraphrased_data))

original_data = load_folder("./train_data")
preprocessor = Preprocessor()
plagiarized_texts, _ = preprocessor.categorize_sentences(original_data)

print(len(plagiarized_texts))

labels_0 = [
    0 for _ in range(50)
] 

labels_1 = [
    1 for _ in range(60)

]

labels = labels_0 + labels_1

# Combine original and plagiarized texts along with their labels
all_texts = original_texts + plagiarized_texts + paraphrased_data
all_labels = labels + [1] * len(plagiarized_texts)

# Split the combined data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(all_texts,
                                                                    all_labels,
                                                                    test_size=0.2,
                                                                    random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2 labels: plagiarized or not

# Prepare datasets and data loaders
train_dataset = PlagiarismDataset(train_texts, train_labels, tokenizer, max_length=128)
val_dataset = PlagiarismDataset(val_texts, val_labels, tokenizer, max_length=128)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Training settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Fine-tune the model
fine_tune_model(model, train_loader, val_loader, optimizer, num_epochs=3)


