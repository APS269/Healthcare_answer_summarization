# !pip install --quiet bert-score rouge-score evaluate sentence-transformers
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from rouge_score import rouge_scorer  # Updated for ROUGE
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
import numpy as np
import pandas as pd
import evaluate  # Use this instead of load_metric

# ----- Constants -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# ----- Load model -----
base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to(device)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = PeftModel.from_pretrained(base_model, "/kaggle/working/summarizer_epoch3_loss3.2330").to(device) 

# ----- Load data -----
with open("/kaggle/input/healthcare/preprocessed_test_data.json", "r") as f:
    test_data = json.load(f)

# ----- Create prompt -----
def create_prompt(entry):
    tone = tone_map[entry['Perspective']]
    anchor = anchor_map[entry['Perspective']]
    return f"""Summarize the following content according to perspective: {entry['Perspective']}
Begin Summary With: {anchor}
Tone: {tone}
Question: {entry['question']}
Content:
""" + "\n".join(entry["answers"])

# ----- Generate summaries -----
results = []
for entry in tqdm(test_data, desc="Generating Summaries"):
    prompt = create_prompt(entry)
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=128)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    results.append({
        "question": entry["question"],
        "perspective": entry["Perspective"],
        "prediction": summary,
        "reference": entry["Summary"]
    })

# Save predictions
os.makedirs("generated", exist_ok=True)
with open("generated/generated_result.json", "w") as f:
    json.dump(results, f, indent=2)

# Save CSV for evaluation
df = pd.DataFrame(results)
df.to_csv("generated/generated_result.csv", index=False)

# ----- Evaluation -----
rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
meteor_metric = evaluate.load("meteor")  # Updated from load_metric
bleu_scores = {"bleu1": [], "bleu2": [], "bleu3": [], "bleu4": []}
predictions = df["prediction"].tolist()
references = df["reference"].tolist()

# ROUGE
rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
for pred, ref in zip(predictions, references):
    scores = rouge_scorer.score(ref, pred)
    for key in rouge_scores:
        rouge_scores[key].append(scores[key].fmeasure)

# METEOR
meteor = meteor_metric.compute(predictions=predictions, references=references)["meteor"]

# BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoothie = SmoothingFunction().method4
for pred, ref in zip(predictions, references):
    ref_tokens = [ref.split()]
    pred_tokens = pred.split()
    bleu_scores["bleu1"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie))
    bleu_scores["bleu2"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie))
    bleu_scores["bleu3"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie))
    bleu_scores["bleu4"].append(sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie))

# BERTScore
P, R, F1 = bert_score(predictions, references, lang="en")

# ----- Print Results -----
print("=== Evaluation Results ===")
print("ROUGE:", {k: np.mean(v) * 100 for k, v in rouge_scores.items()})
print("METEOR:", meteor)
print("BLEU:", {k: np.mean(v) * 100 for k, v in bleu_scores.items()})
print("BERTScore F1:", F1.mean().item() * 100)
