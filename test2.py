import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

def load_data_from_folder(folder_path):
    data = []

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder path {folder_path} does not exist.")

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8-sig") as f:
                data.append(f.read())
    return data

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
        print(classification_report(val_labels, val_preds))

# Example data
preprocessor = Preprocessor()

unordered_texts = load_data_from_folder("./generated_data/unordered_data/")
unordered_texts, _ = preprocessor.categorize_sentences(unordered_texts)

unordered_texts = unordered_texts

paraphrased_texts = load_data_from_folder("./generated_data/paraphrase_data/")
paraphrased_texts, _ = preprocessor.categorize_sentences(paraphrased_texts)

paraphrased_texts = paraphrased_texts

original_texts = load_data_from_folder("./train_data")
original_texts, _ = preprocessor.categorize_sentences(original_texts)

# Combine original, paraphrased, and plagiarized texts along with their labels
all_texts = unordered_texts + paraphrased_texts + original_texts
all_labels = [0] * len(unordered_texts) + [1] * len(paraphrased_texts) + [2] * len(original_texts)

# Split the combined data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(all_texts,
                                                                    all_labels,
                                                                    test_size=0.2,
                                                                    random_state=42)

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)  # 3 labels: original, paraphrased, plagiarized

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
fine_tune_model(model, train_loader, val_loader, optimizer, num_epochs=5)
