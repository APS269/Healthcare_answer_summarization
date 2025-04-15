# Healthcare Answer Summarization

This repository contains code for training and evaluating a classifier and a summarizer model to generate healthcare-related summaries based on user queries and perspectives.

## 🔧 Project Structure

```
.
├── baseline1/
├── baseline2/
├── classifier/
│   └── best_classifier.ckpt
├── summarizer/
├── classify.py
├── summarize.py
├── inference.py
├── train.json
├── valid.json
├── test.json
├── preprocessed_all_data.json
├── preprocessed_valid_data.json
├── preprocessed_test_data.json
├── final_Eval_metrics_test.png
├── classifier_9epochs.png
├── classifier_9_ahead_epochs.png
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/APS269/Healthcare_answer_summarization.git
cd Healthcare_answer_summarization
```

### 2. Install dependencies

Make sure you have Python 3.8+ and the required libraries installed.

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install torch transformers peft sentence-transformers tqdm matplotlib evaluate bert-score rouge-score
```

### 3. Run the classifier

To train the classifier model:

```bash
python classify.py
```

This will train a RoBERTa-based model for classifying summaries into five perspectives:
- EXPERIENCE
- SUGGESTION
- INFORMATION
- CAUSE
- QUESTION

The best model checkpoint will be saved at:  
`classifier/best_classifier.ckpt`

### 4. Run the summarizer

To train the summarizer model:

```bash
python summarize.py
```

This uses FLAN-T5 with prefix tuning and a custom energy-based loss for perspective-aligned summarization.


### 5. Run inference and evaluate

## 🔍 Inference + Evaluation

Run on the test set to generate summaries and evaluate them.

After training the summarizer, run:

```bash
python inference.py
```


### Output:

- Generated summaries: `generated/generated_result.csv`
- Evaluation scores: `generated/eval_scores.json`

Metrics reported:
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**
- **METEOR**
- **BERTScore F1**

---

## 📊 Sample Evaluation Output

```
=== 🮾 Evaluation Metrics ===
🔸 METEOR: 5.63
🔸 BLEU: {'bleu1': 2.51, 'bleu2': 1.12, 'bleu3': 0.64, 'bleu4': 0.39}
🔸 BERTScore F1: 83.68
```

---

## 📌 Notes

- `train_summarizer.py` only uses cross-entropy for training (no energy-based loss).
- `inference_eval.py` is standalone and does inference + full evaluation.
- Make sure all model and data paths are consistent.


This script:
- Generates summaries on test data
- Computes ROUGE, METEOR, BLEU, and BERTScore metrics
- Saves outputs to `generated/` folder

---

## 📁 Submission

```
79_endevaluation.zip
```


The zip file must include:
- one baseline (one was submitted before during mid-evaluation)
- All code files (`.py`)
- Trained model checkpoints
- Evaluation plots & output files

---




