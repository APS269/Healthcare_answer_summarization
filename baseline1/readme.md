# Healthcare Answer Summarization

This project focuses on generating perspective-based summaries of healthcare-related answers using a fine-tuned sequence-to-sequence model with prefix tuning. The goal is to optimize BLEU, ROUGE, and BERTScore metrics.

## Features

- Uses **Flan-T5** for text summarization (can be swapped with BART).
- **Prefix Tuning** applied for efficient parameter tuning.
- Custom **Perspective Alignment Loss** for enforcing perspective correctness.
- Evaluation with **BLEU, ROUGE, and BERTScore**.
- Supports **tokenization, training, inference, and evaluation**.

---

## Installation

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- TQDM
- NLTK
- Rouge Score
- BERTScore

### Install Dependencies

```bash
pip install torch transformers datasets peft tqdm nltk rouge-score bert-score
```

---

## Dataset Format

The dataset should be in JSON format:

```json
[
  {
    "question": "What are the benefits of turmeric?",
    "answers": [
      "Turmeric has anti-inflammatory properties.",
      "It helps with digestion."
    ],
    "labelled_summaries": {
      "INFORMATION": "Turmeric is known for its anti-inflammatory and digestive benefits.",
      "SUGGESTION": "Consider adding turmeric to your diet for its health benefits."
    }
  }
]
```

---

## Training & Evaluation

### 1️ Data Preprocessing

```python
python preprocess.py --input train.json --output train_processed.json
```

### 2️ Train Model

```python
python train.py --epochs 3 --batch_size 8 --lr 1e-5
```

### 3️ Evaluate Model

```python
python evaluate.py --model checkpoint.pth --data val.json
```

---

## Metrics

- **BLEU Score**: Measures how closely generated text matches the reference.
- **ROUGE-L**: Measures recall and precision of generated summaries.
- **BERTScore**: Computes similarity between generated and reference summaries.

---

## Inference

```python
python infer.py --model checkpoint.pth --input "How does vitamin C boost immunity?"
```

Example Output:

```
"Vitamin C supports the immune system by reducing oxidative stress."
```

---

## Performance Results

| Model                   | BLEU | ROUGE-L | BERTScore |
| ----------------------- | ---- | ------- | --------- |
| Flan-T5 (Baseline)      | 0.65 | 0.72    | 0.81      |
| Flan-T5 + Prefix Tuning | 0.72 | 0.78    | 0.85      |
