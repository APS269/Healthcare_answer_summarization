import json
import re
import torch
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
    AdamW, get_scheduler
)
from peft import PrefixTuningConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def clean_text(text):
    text = re.sub(r"[\n\\]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    formatted_data = []
    for entry in data:
        question = clean_text(entry["question"])
        answers = " ".join(clean_text(a) for a in entry["answers"])
        for perspective, summary in entry["labelled_summaries"].items():
            input_text = f"Summarize the following healthcare answer from a {perspective} perspective: {question} {answers}"
            formatted_data.append({
                "input_text": input_text,
                "summary": clean_text(summary),
                "perspective": perspective
            })
    return Dataset.from_list(formatted_data).train_test_split(test_size=0.1)

MODEL_NAME = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"], 
        padding="max_length", truncation=True, max_length=512
    )
    labels = tokenizer(
        examples["summary"], 
        padding="max_length", truncation=True, max_length=150
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_data("train.json")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "summary"])
train_dataset, val_dataset = tokenized_dataset["train"], tokenized_dataset["test"]
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME, padding=True)

def custom_collate_fn(features):
    perspectives = [feature.pop("perspective") for feature in features]
    batch = data_collator(features)
    batch["perspective"] = perspectives
    return batch

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
peft_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    num_virtual_tokens=8,
    token_dim=model.config.d_model
)
peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
device = "cuda" if torch.cuda.is_available() else "cpu"
peft_model.to(device)

def compute_loss(model, input_ids, attention_mask, labels, perspective):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    ce_loss = outputs.loss
    ep_loss = perspective_alignment_loss(outputs.logits, perspective)
    return ce_loss + ep_loss

def perspective_alignment_loss(logits, perspective):
    softmax_probs = F.softmax(logits, dim=-1)
    target_idx = {"INFORMATION": 0, "SUGGESTION": 1, "EXPERIENCE": 2, "CAUSE": 3, "QUESTION": 4}
    if isinstance(perspective, list):
        perspective = perspective[0]
    if perspective in target_idx:
        target_prob = softmax_probs[:, :, target_idx[perspective]].mean()
        loss = -torch.log(target_prob + 1e-10)
        return loss
    return torch.tensor(0.0, device=device)

optimizer = AdamW(peft_model.parameters(), lr=1e-5)
num_training_steps = len(train_dataloader) * 3
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=1000, num_training_steps=num_training_steps)

for epoch in range(3):
    peft_model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        perspective = batch["perspective"][0] if isinstance(batch["perspective"], list) else batch["perspective"]
        optimizer.zero_grad()
        loss = compute_loss(peft_model, input_ids, attention_mask, labels, perspective)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} - Avg Train Loss: {avg_train_loss:.4f}")

def evaluate(model, dataloader):
    model.eval()
    all_predictions, all_references = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model.base_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100)
            generated_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            reference_summaries = tokenizer.batch_decode(labels, skip_special_tokens=True)
            all_predictions.extend(generated_summaries)
            all_references.extend(reference_summaries)
    bleu = np.mean([sentence_bleu([ref], pred) for ref, pred in zip(all_references, all_predictions)])
    rouge = rouge_scorer.RougeScorer(["rougeL"]).score(all_references[0], all_predictions[0])["rougeL"].fmeasure
    _, _, bert = bert_score(all_predictions, all_references, lang="en", rescale_with_baseline=True)
    print(f"BLEU: {bleu:.4f}, ROUGE-L: {rouge:.4f}, BERTScore: {np.mean(bert):.4f}")

evaluate(peft_model, val_dataloader)
