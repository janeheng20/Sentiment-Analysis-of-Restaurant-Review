import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU Name:", torch.cuda.get_device_name(0))

# Load data
data2 = pd.read_csv("Augmented_Sentiment_data.csv")

# Preprocessing
train_texts, test_texts, train_labels, test_labels = train_test_split(
    data2['Review'].astype(str).tolist(),
    data2['Sentiment'].tolist(),
    test_size=0.2,
    stratify=data2['Sentiment'],
    random_state=42
)

# Tokenization
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

# Dataset class
class ReviewDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = ReviewDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic model training
model_basic = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)
optimizer = AdamW(model_basic.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss()

print("Training Basic Model...")
model_basic.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model_basic(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Basic Model Loss: {total_loss/len(train_loader):.4f}")

model_basic.save_pretrained("distilbert_basic_sentiment_model")
tokenizer.save_pretrained("distilbert_basic_sentiment_model")

# Weighted model training
class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
model_weighted = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)
optimizer = AdamW(model_weighted.parameters(), lr=2e-5)
loss_fn = CrossEntropyLoss(weight=class_weights)

print("Training Weighted Model...")
model_weighted.train()
for epoch in range(3):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model_weighted(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Weighted Model Loss: {total_loss/len(train_loader):.4f}")

model_weighted.save_pretrained("distilbert_weighted_sentiment_model")
tokenizer.save_pretrained("distilbert_weighted_sentiment_model")
print("Training complete and models saved.")
