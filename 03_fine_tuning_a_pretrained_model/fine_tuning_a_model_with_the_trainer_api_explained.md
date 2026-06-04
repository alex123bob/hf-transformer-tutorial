# fine_tuning_a_model_with_the_trainer_api.py — Annotated Walkthrough

> **What this script does:** Fine-tunes BERT on a paraphrase detection task using HuggingFace's `Trainer` — the high-level API that handles the entire training loop for you.

```
┌─────────────── TRAINER vs MANUAL LOOP ──────────────────────────┐
│                                                                   │
│  WITH Trainer (this script):      WITHOUT Trainer (next script): │
│                                                                   │
│  trainer = Trainer(...)           for epoch in range(3):         │
│  trainer.train()  ← that's it!     for batch in loader:          │
│                                       outputs = model(**batch)   │
│  ~10 lines total                      loss.backward()            │
│                                       optimizer.step()           │
│                                       scheduler.step()           │
│                                       optimizer.zero_grad()      │
│                                    …40+ lines                    │
│                                                                   │
│  Trainer gives you FREE:                                          │
│  ✓ Mixed precision  ✓ Checkpointing  ✓ Multi-GPU                 │
│  ✓ Logging  ✓ Evaluation  ✓ Progress bars                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
import torch                                                      # 4
import numpy as np                                                # 5
import evaluate                                                   # 6
from torch.optim import AdamW                                     # 7
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer  # 8
from datasets import load_dataset                                 # 10
```
> **Line 1-10:** New imports vs previous scripts:
>
> | Import | Role |
> |--------|------|
> | `evaluate` | Library for computing metrics (accuracy, F1) |
> | `TrainingArguments` | Config object — epochs, batch size, save path, etc. |
> | `Trainer` | High-level training loop handler |

---

## Data Preparation

```python
raw_datasets = load_dataset("glue", "mrpc")                       # 12
checkpoint = "bert-base-uncased"                                  # 13
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 14
```
> **Line 12-14:** Load MRPC paraphrase dataset and BERT tokenizer.

```python
def tokenize_function(examples):                                  # 16
    print(f"examples structure: {repr(examples)}")                # 17
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)  # 18
```
> **Line 16-18:** Tokenize sentence pairs, truncate if over 512. No padding here — DataCollator handles that per batch.

```python
print(f"raw datasets structure: {raw_datasets}")                  # 20
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 21
print(f"tokenized datasets structure: {tokenized_datasets}")      # 22
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)      # 23
```
> **Line 20-23:** Apply tokenization across all splits, create dynamic-padding collator.

---

## Training Configuration

```python
training_args = TrainingArguments('dist/models', eval_strategy="epoch")  # 25
```
> **Line 25:** Configure training — just two explicit args here, but many defaults apply:
> ```
> ┌─── TrainingArguments defaults ────────────────────────────────┐
> │                                                               │
> │  'dist/models'         → save checkpoints here               │
> │  eval_strategy="epoch" → evaluate after EACH epoch           │
> │                                                               │
> │  Implicit defaults:                                           │
> │  num_train_epochs              = 3                            │
> │  per_device_train_batch_size   = 8                            │
> │  learning_rate                 = 5e-5                         │
> │  weight_decay                  = 0                            │
> │  warmup_steps                  = 0                            │
> │  logging_steps                 = 500                          │
> │                                                               │
> │  Any of these can be overridden:                              │
> │  TrainingArguments('dist/models', num_train_epochs=5,         │
> │                    learning_rate=3e-5, ...)                   │
> └───────────────────────────────────────────────────────────────┘
> ```

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)  # 26
```
> **Line 26:** Load BERT with a 2-class head (paraphrase = 1, not paraphrase = 0):
> ```
> ┌─── What is "fine-tuning"? ─────────────────────────────────────┐
> │                                                                │
> │  Pre-trained BERT: understands English grammar, world facts    │
> │                    but knows nothing about our task            │
> │                                                                │
> │  Fine-tuning: show it our labeled examples, let it adapt       │
> │                                                                │
> │  ┌──────────────────────────────────────────────────────────┐  │
> │  │  Pre-trained weights  +  Classification head              │  │
> │  │  (good at language)      (randomly initialized)           │  │
> │  │        ↓  fine-tuning trains both                         │  │
> │  │  Slightly tweaked      Heavily trained on our task        │  │
> │  └──────────────────────────────────────────────────────────┘  │
> │                                                                │
> │  Analogy: hire a fluent English speaker (pre-trained BERT),   │
> │  then teach them your company's specific job (fine-tuning).   │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Custom Evaluation Metrics

```python
def compute_metrics(eval_preds):                                  # 28
    metric = evaluate.load("glue", "mrpc")                        # 29
    logits, labels = eval_preds                                   # 30
    predictions = np.argmax(logits, axis=-1)                      # 31
    print('running compute_metrics')                              # 32
    return metric.compute(predictions=predictions, references=labels)  # 33
```
> **Line 28-33:** Called by Trainer after each evaluation epoch:
> ```
> ┌─── compute_metrics step by step ──────────────────────────────┐
> │                                                                │
> │  eval_preds = (logits, labels)                                 │
> │    logits shape: [N, 2]  — raw model scores for all N samples  │
> │    labels shape: [N]     — correct answers (0 or 1)            │
> │                                                                │
> │  np.argmax(logits, axis=-1):                                   │
> │    [[0.2, 0.8], [0.9, 0.1], …]                                 │
> │    →  [1,        0,         …]   ← index of the higher score   │
> │                                                                │
> │  metric.compute():                                             │
> │    MRPC standard metrics → {"accuracy": 0.85, "f1": 0.89}     │
> │                                                                │
> │  F1 = 2 × (precision × recall) / (precision + recall)         │
> │  Better than accuracy when classes are imbalanced.             │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Assembling and Running the Trainer

```python
trainer = Trainer(                                                # 35
    model,                                                        # 36
    training_args,                                                # 37
    train_dataset=tokenized_datasets["train"],                    # 38
    eval_dataset=tokenized_datasets["validation"],                # 39
    data_collator=data_collator,                                  # 40
    processing_class=tokenizer,                                   # 41
    compute_metrics=compute_metrics                               # 42
)                                                                 # 43
```
> **Line 35-43:** Wire all the pieces together:
> ```
> ┌─── Trainer components ─────────────────────────────────────────┐
> │                                                                │
> │  model            → the neural network to train                │
> │  training_args    → hyperparameters (epochs, lr, save path)   │
> │  train_dataset    → 3668 training samples                      │
> │  eval_dataset     → 408 validation samples                     │
> │  data_collator    → dynamic padding per batch                  │
> │  processing_class → tokenizer (for saving/logging)            │
> │  compute_metrics  → called after each epoch to print scores   │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
trainer.train()                                                   # 45
```
> **Line 45:** Start training. This single call runs the entire loop:
> ```
> ┌─── What trainer.train() does internally ───────────────────────┐
> │                                                                │
> │  For each epoch (1, 2, 3):                                     │
> │  │                                                             │
> │  │  For each batch of 8 samples (458 batches per epoch):       │
> │  │  │  1. data_collator pads batch to its max length           │
> │  │  │  2. Move batch to GPU/CPU                                │
> │  │  │  3. model(**batch) → loss                                │
> │  │  │  4. loss.backward() → gradients                          │
> │  │  │  5. optimizer.step() → update weights                    │
> │  │  │  6. scheduler.step() → adjust learning rate              │
> │  │  │  7. Log loss every 500 steps                             │
> │  │                                                             │
> │  │  After epoch: evaluate on validation set                    │
> │  │  Print: {"eval_accuracy": 0.84, "eval_f1": 0.88}           │
> │                                                                │
> │  Total steps: 3 epochs × 458 batches = 1374 steps             │
> │  Typical time: ~5 min GPU, ~30 min CPU                         │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Training Progress Visualised

```
┌─── Loss and Accuracy Over Training ────────────────────────────┐
│                                                                │
│  loss                                                          │
│  0.7 ┤╲                                                        │
│  0.5 ┤  ╲╲                                                     │
│  0.3 ┤     ╲──╲                                                │
│  0.1 ┤         ╲────                                           │
│      └────┬────┬────┬──→ epochs                                │
│           1    2    3                                          │
│                                                                │
│  accuracy                                                      │
│  0.9 ┤         ╱────                                           │
│  0.8 ┤     ╱──╱                                                │
│  0.7 ┤  ╱╱                                                     │
│  0.6 ┤╱                                                        │
│      └────┬────┬────┬──→ epochs                                │
│           1    2    3                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Fine-tuning** | Adapt a pre-trained model to a specific task with your data |
| **Epoch** | One full pass through all training data |
| **TrainingArguments** | All training hyperparameters in one config object |
| **Trainer** | Handles batching, forward/backward, logging, saving — everything |
| **eval_strategy** | When to evaluate: `"epoch"` = after each epoch, `"steps"` = every N steps |
| **compute_metrics** | Custom function called after each eval to compute accuracy/F1 |
| **argmax** | Pick the class index with the highest score |
| **F1 score** | Harmonic mean of precision and recall — better than accuracy for imbalanced data |
