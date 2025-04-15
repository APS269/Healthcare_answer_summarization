import torch
import json
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from peft import get_peft_model, PrefixTuningConfig, TaskType
from sentence_transformers import SentenceTransformer, util
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import math
import os
## Creating Classifier Dataset Class
class classifierDataset(Dataset):
    
    def __init__(self , data  , tokenizer , max_length = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"EXPERIENCE": 0, "SUGGESTION": 1, "INFORMATION": 2, "CAUSE": 3, "QUESTION": 4}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        text = self.data[index]['Summary']
        label = self.label_map[self.data[index]['Perspective']]
        
        encoded = self.tokenizer(
            text, 
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt')
            
        return {
            'input_ids' : encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "label": torch.tensor(label)
        }


## Assigning the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Using RoBerta tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5).to(device)
batch_size = 4

# Path to files
train_path = r'L:\79_endevaluation\preprocessed_all_data.json'
val_path = r"L:\79_endevaluation\preprocessed_valid_data.json"

# Opening files
with open(train_path) as f:
    train_data = json.load(f)
with open(val_path) as f:
    val_data = json.load(f)
    
# Creating Dataset object  
train_dataset = classifierDataset(
    data=train_data,
    tokenizer=tokenizer)
val_dataset = classifierDataset(
    data=val_data,
    tokenizer=tokenizer)

# creating data Loader Object
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size
    )


## Hyper Parameters and model optimizer
num_epochs = 15
learning_rate = 0.0001

optimizer = AdamW(model.parameters() , lr=learning_rate)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')

## Directory to save classifier model
cls_dir = "classifier/best_classifier.ckpt"

## Training loop

for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader , desc=f"Training Epoch {epoch+1}"):
        
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch['label'].to(device)
        
        output = model(input_ids=input_ids , attention_mask=attention_mask , labels=labels)
        loss = output.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_train_loss = total_loss/len(train_loader)
    print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}")
    
    # Save best model
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        
        for batch in val_loader:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            val_loss += outputs.loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        os.makedirs(os.path.dirname(cls_dir) , exist_ok=True)
        torch.save({
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()
            }, cls_dir)
        print('CheckPoint save')