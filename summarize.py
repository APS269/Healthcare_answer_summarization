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
from torchmetrics.text import BLEUScore, BERTScore



perspective_labels = ["EXPERIENCE", "SUGGESTION", "INFORMATION", "CAUSE", "QUESTION"]
cls_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
cls_model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
cls_checkpoint = torch.load("classifier/best_classifier.ckpt")
cls_model.load_state_dict(cls_checkpoint['model_state_dict'])
cls_model.eval().cuda()


embedder = SentenceTransformer("all-MiniLM-L6-v2")

tone_map = {
    "SUGGESTION": "Advisory, Recommending",
    "INFORMATION": "Informative, Educational",
    "CAUSE": "Explanatory, Causal",
    "EXPERIENCE": "Personal, Narrative",
    "QUESTION": "Seeking Understanding"
}

anchor_map = {
    "SUGGESTION": "It is suggested",
    "INFORMATION": "For information purposes",
    "CAUSE": "Some of the causes",
    "EXPERIENCE": "In user's experience",
    "QUESTION": "It is inquired"
}


## Assigning the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base').to(device)
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')

peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_2_SEQ_LM , num_virtual_tokens=20 , prefix_projection=True)
model = get_peft_model(summarizer_model, peft_config).to("cuda")
model.print_trainable_parameters()


def compute_Ep(summary, true_label):
    inputs = cls_tokenizer(summary, return_tensors="pt", padding=True, truncation=True).to("cuda")
    logits = cls_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0][perspective_labels.index(true_label)].item()

def compute_Ea(summary, perspective):
    anchor = anchor_map[perspective].lower()
    return 1.0 if summary.lower().startswith(anchor) else 0.0

def compute_Et(summary, perspective):
    tone = tone_map[perspective]
    emb_sum = embedder.encode(summary, convert_to_tensor=True)
    emb_tone = embedder.encode(tone, convert_to_tensor=True)
    return util.cos_sim(emb_sum, emb_tone).item()


def energy_loss(summary, true_label):
    alpha, beta, gamma = 0.7, 0.3, 0.5
    Ep_val = compute_Ep(summary, true_label)
    Ea_val = compute_Ea(summary, true_label)
    Et_val = compute_Et(summary, true_label)

    E = {}
    for label in perspective_labels:
        E[label] = alpha * compute_Ep(summary, label) + beta * compute_Ea(summary, label) + gamma * compute_Et(summary, label)

    exp_E = {k: math.exp(-1 / v) for k, v in E.items()}
    Z = sum(exp_E.values())
    P = {k: v / Z for k, v in exp_E.items()}

    Y = {k: 1.0 if k == true_label else 0.0 for k in perspective_labels}
    P_tensor = torch.tensor([P[k] for k in perspective_labels])
    Y_tensor = torch.tensor([Y[k] for k in perspective_labels])
    loss = -torch.sum(Y_tensor * torch.log(P_tensor + 1e-8))
    return loss


class summarizeDataset(Dataset):
    
    def __init__(self , data , tokenizer , max_length= 512 , target_length= 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_length = target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        item = self.data[index]
        
        input_encoded = self.tokenizer(item['input'] ,
                                       max_length= self.max_length ,
                                       padding= 'max_length' ,
                                       truncation=True,
                                       return_tensors='pt')
        target_encoded = self.tokenizer(item['target'] ,
                                        max_length= self.target_length ,
                                        padding='max_length' , 
                                        truncation=True,
                                        return_tensors='pt')
        
        return {
            "input_ids": input_encoded["input_ids"].squeeze(),
            "attention_mask": input_encoded["attention_mask"].squeeze(),
            "labels": target_encoded["input_ids"].squeeze(),
            "perspective": item["Perspective"]           
        }
        



# Path to files
train_path = r'L:\79_endevaluation\preprocessed_all_data.json'
val_path = r"L:\79_endevaluation\preprocessed_valid_data.json"

# Opening files
with open(train_path) as f:
    train_data = json.load(f)
with open(val_path) as f:
    val_data = json.load(f)
    

def prompt_input(input):
    
    tone = tone_map[input['Perspective']]
    anchor = anchor_map[input['Perspective']]
    
    input_prompt_text = f"""Summarize the following content according to perspective: {input["Perspective"]}
                        Begin Summary With: {anchor}
                        Tone: {tone}
                        Question: {input["question"]}
                        Content:
                        """ + "\n".join(input["answers"])
    
    return {
        'input' : input_prompt_text,
        'target' : input["Summary"] ,
        'Perspective' : input['Perspective']
    }

train_prompt_data = [prompt_input(input) for input in train_data]
val_prompt_data = [prompt_input(input) for input in val_data]

train_dataset = summarizeDataset(train_prompt_data , tokenizer=tokenizer)
val_dataset = summarizeDataset(val_prompt_data , tokenizer=tokenizer)

train_dataloader = DataLoader(train_dataset , batch_size=16 , shuffle=True)
val_dataloader = DataLoader(val_dataset , batch_size=16)





for batch in train_dataloader:
    print(f"{batch['input_ids']} and {batch['input_ids'].shape}")
    print(f"{batch['attention_mask']} and  {batch['attention_mask'].shape}")
    print(f"{batch['labels']} and {batch['labels'].shape}")
    break



num_epochs = 4
learning_rate = 0.0001

optimizer = AdamW(model.parameters() , lr=learning_rate)
criterion = nn.CrossEntropyLoss()

best_val_loss =  float('inf')

for epoch in range(num_epochs):
    
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_dataloader , desc=f'Training epoch {epoch+1}'):
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels =  batch['labels'].to(device)
        
        output = model(input_ids=input_ids , attention_mask=attention_mask , labels=labels)
        loss = output.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")   
    
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        
        for batch in val_dataloader:
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels =  batch['labels'].to(device)

            output = model(input_ids=input_ids , attention_mask=attention_mask , labels=labels)
            generate_summary = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128)
            decoded_summaries = tokenizer.batch_decode(generate_summary, skip_special_tokens=True)

            energy_losses = [energy_loss(summary, perspective) for summary, perspective in zip(decoded_summaries, batch["perspective"])]
            energy_total = sum(energy_losses) / len(energy_losses)
            
            val_loss += (output.loss.item() + energy_total.item())
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(f"summarizer/best_model_epoch{epoch+1}_loss{avg_val_loss:.4f}")
        print(" Saved best model.")