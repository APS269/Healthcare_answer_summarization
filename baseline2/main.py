import json
import torch
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
)

# ------------ Load Preprocessed Test Data ------------
def load_preprocessed_test_data(file_path):
    with open(file_path, "r") as f:
        raw_data = json.load(f)
    data = []
    for item in raw_data:
        question = item["question"]
        answers = " ".join(item["answers"])
        perspective = item["Perspective"]
        summary = item["Summary"]
        input_text = f"Summarize the following healthcare answer from a {perspective} perspective: {question} {answers}"
        data.append({
            "input_text": input_text,
            "summary": summary
        })
    return data

# ------------ Tokenize and Create Dataloader ------------
def prepare_dataloader(data, model_name, batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(example):
        inputs = tokenizer(example["input_text"], padding="max_length", truncation=True, max_length=512)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(example["summary"], padding="max_length", truncation=True, max_length=150)
        inputs["labels"] = labels["input_ids"]
        return inputs

    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_fn, remove_columns=["input_text", "summary"])
    collator = DataCollatorForSeq2Seq(tokenizer, model=model_name)
    return DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator), tokenizer

# ------------ Evaluation Function ------------
def evaluate(model_name, val_loader, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    all_preds, all_refs = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=100)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            refs = tokenizer.batch_decode(labels, skip_special_tokens=True)

            all_preds.extend(preds)
            all_refs.extend(refs)

    # BLEU Score
    bleu = np.mean([sentence_bleu([ref], pred) for ref, pred in zip(all_refs, all_preds)])

    # ROUGE-L Score
    rouge_scores = [
        rouge_scorer.RougeScorer(["rougeL"]).score(ref, pred)["rougeL"].fmeasure
        for ref, pred in zip(all_refs, all_preds)
    ]
    rouge = np.mean(rouge_scores)

    # BERTScore (Precision, Recall, F1)
    _, _, bert_f1 = bert_score(all_preds, all_refs, lang="en", rescale_with_baseline=True)
    bert_f1_mean = bert_f1.mean().item()

    # Print all scores
    print(f"BLEU: {bleu:.4f}")
    print(f"ROUGE-L: {rouge:.4f}")
    print(f"BERTScore (F1): {bert_f1_mean:.4f}")

# ------------ Main Run ------------
if __name__ == "__main__":
    model_name = "facebook/bart-base"  # Or any other Seq2Seq model
    file_path = r"L:\79_endevaluation\preprocessed_test_data.json"
    
    test_data = load_preprocessed_test_data(file_path)
    val_loader, tokenizer = prepare_dataloader(test_data, model_name)
    evaluate(model_name, val_loader, tokenizer)
