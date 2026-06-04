# processing_the_data.py — Annotated Walkthrough

> **What this script does:** Covers how to prepare real dataset for training — loading, tokenizing, and batching data efficiently. This is the critical step before any training happens.

```
┌─────────────────── THE DATA PIPELINE ───────────────────────────┐
│                                                                   │
│  Raw text dataset                                                 │
│       ↓  load_dataset()                                           │
│  HuggingFace Dataset object                                       │
│       ↓  dataset.map(tokenize_function)                           │
│  Tokenized dataset (numbers, no padding yet)                      │
│       ↓  DataCollatorWithPadding                                  │
│  Dynamically padded batches   ← ready for training!               │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
import torch                                                      # 4
from torch.optim import AdamW                                     # 5
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding  # 6
from datasets import load_dataset                                 # 8
```
> **Line 1-8:** Imports:
>
> | Import | Role |
> |--------|------|
> | `AdamW` | Optimizer — adjusts model weights during training |
> | `DataCollatorWithPadding` | Groups samples into padded batches dynamically |
> | `load_dataset` | Downloads/loads datasets from HuggingFace Hub |

---

## Part 1: A Mini Training Step (Concept Preview)

```python
checkpoint = "bert-base-uncased"                                  # 10
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 11
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # 12
```
> **Line 10-12:** Load BERT and its tokenizer. No `num_labels` specified here — defaults to 2.

```python
sequences = [                                                     # 14
    "I've been waiting for a HuggingFace course my whole life.",  # 15
    "This course is amazing!",                                    # 16
]                                                                 # 17
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")  # 18
```
> **Line 14-18:** Tokenize 2 sentences into a padded batch tensor.

```python
# This is new
batch["labels"] = torch.tensor([1, 1])                           # 21
```
> **Line 21:** Add the ground-truth labels to the batch:
> ```
> ┌─── What are "labels"? ─────────────────────────────────────────┐
> │                                                                │
> │  Labels = the CORRECT answers we want the model to learn.      │
> │                                                                │
> │  label=1 means "POSITIVE", label=0 means "NEGATIVE"            │
> │  Both sentences here are positive → [1, 1]                    │
> │                                                                │
> │  This is supervised learning — we explicitly tell the model    │
> │  the right answer, then measure how wrong it was (= loss).     │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
optimizer = AdamW(model.parameters())                             # 23
```
> **Line 23:** Create the optimizer — the algorithm that updates model weights after each batch:
> ```
> ┌─── What is an Optimizer? ──────────────────────────────────────┐
> │                                                                │
> │  BERT has ~110M parameters (just numbers).                     │
> │  Training = adjusting these numbers so predictions improve.    │
> │                                                                │
> │  AdamW decides HOW to adjust each weight:                      │
> │  • Direction: which way does this weight need to move?         │
> │  • Magnitude: how big a step?                                  │
> │                                                                │
> │  model.parameters() gives the optimizer access to all weights. │
> │  "W" in AdamW = weight decay (penalises very large weights     │
> │   to prevent overfitting).                                     │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
loss = model(**batch).loss                                        # 24
```
> **Line 24:** Forward pass — model predicts AND computes how wrong it is:
> ```
> ┌─── Loss Explained ─────────────────────────────────────────────┐
> │                                                                │
> │  When batch includes "labels", the model automatically        │
> │  computes cross-entropy loss:                                  │
> │                                                                │
> │  Prediction: [0.3, 0.7] → 70% confident it's POSITIVE         │
> │  Label:      1           → correct answer IS POSITIVE          │
> │  Loss:       0.36        → not too wrong                       │
> │                                                                │
> │  Prediction: [0.9, 0.1] → 90% confident it's NEGATIVE         │
> │  Label:      1           → correct answer IS POSITIVE          │
> │  Loss:       2.30        → very wrong!                         │
> │                                                                │
> │  Goal of training: drive loss as close to 0 as possible.      │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
loss.backward()                                                   # 25
```
> **Line 25:** **Backpropagation** — compute how each weight should change to reduce loss:
> ```
> ┌─── What is backward()? ────────────────────────────────────────┐
> │                                                                │
> │  Calculates the gradient for every parameter:                  │
> │  "If I nudge this weight up by 0.001, does loss go up or down?"│
> │                                                                │
> │  Analogy: you're standing on a hilly landscape (loss = height) │
> │  backward() tells you the slope in every direction             │
> │  so you know which way is "downhill" (lower loss)              │
> │                                                                │
> │  After backward(), every parameter has a .grad value           │
> │  pointing in the direction that increases loss.                │
> │  The optimizer will step in the OPPOSITE direction.            │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
optimizer.step()                                                  # 26
```
> **Line 26:** Update every weight using the computed gradients:
> ```
> weight_new = weight_old - learning_rate × gradient
>
> ┌─── forward → backward → step ──────────────────────────────────┐
> │                                                                │
> │  model(**batch) → loss    "how wrong am I?"                    │
> │  loss.backward()          "in which direction is improvement?" │
> │  optimizer.step()         "take a step in that direction"      │
> │                                                                │
> │  One loop of these 3 = one training step.                      │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Part 2: Loading a Real Dataset

```python
raw_datasets = load_dataset("glue", "mrpc")                       # 28
raw_train_datasets = raw_datasets["train"]                        # 29
raw_validation_datasets = raw_datasets["validation"]              # 30
```
> **Line 28-30:** Load MRPC (Microsoft Research Paraphrase Corpus):
> ```
> ┌─── MRPC Dataset ───────────────────────────────────────────────┐
> │                                                                │
> │  Task: are these two sentences paraphrases of each other?     │
> │                                                                │
> │  Example:                                                      │
> │  sentence1: "The company said it would restate earnings."      │
> │  sentence2: "The firm said it will restate its earnings."      │
> │  label:     1  (YES, same meaning)                             │
> │                                                                │
> │  Splits:                                                       │
> │    "train":      3668 examples  ← used to train the model      │
> │    "validation":  408 examples  ← used to evaluate during train│
> │    "test":       1725 examples  ← final evaluation             │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
print(raw_train_datasets.features)                                # 32
print(f"15 element of training set: {raw_train_datasets[14]}")    # 34
print(f"87 element of validation set: {raw_validation_datasets[86]}")  # 35
```
> **Line 32-35:** Inspect dataset structure. `.features` shows the column types, indexing `[14]` fetches one sample (0-indexed).

```python
print(f"raw train datasets: {raw_train_datasets['sentence1']} first 10 elements of raw train datasets: {raw_train_datasets['sentence1'][:10]}")  # 37
tokenized_sentence_1_of_training_set = tokenizer(raw_train_datasets["sentence1"][:10])  # 38
tokenized_sentence_2_of_training_set = tokenizer(raw_train_datasets["sentence2"][:10])  # 40
```
> **Line 37-41:** Tokenize the first 10 sentence1s and sentence2s independently — just for inspection.

---

## Part 3: Tokenizing Sentence Pairs

```python
inputs = tokenizer("This is the first sentence.", "This is the second one.")  # 43
print(f"paired inputs after tokenization: {inputs}")              # 44
```
> **Line 43-44:** BERT is designed for sentence pairs — pass both to the tokenizer at once:
> ```
> ┌─── Single vs Pair Tokenization ────────────────────────────────┐
> │                                                                │
> │  Single:  [CLS] sentence A [SEP]                               │
> │                                                                │
> │  Pair:    [CLS] sentence A [SEP] sentence B [SEP]              │
> │           ├──── token_type_ids=0 ──┤├── token_type_ids=1 ──┤   │
> │                                                                │
> │  token_type_ids distinguishes sentence A (0s) from B (1s).    │
> │  This tells BERT: "compare these two sentences."               │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
sentence_1 = raw_train_datasets[14]["sentence1"]                  # 47
sentence_2 = raw_train_datasets[14]["sentence2"]                  # 48
tokenized_sentence_1 = tokenizer(sentence_1, sentence_2)          # 51
print(f"Decode tokenized sentence 1 and 2 as a pair: {tokenizer.decode(tokenized_sentence_1['input_ids'])}")  # 53
```
> **Line 47-53:** Tokenize a real pair from the dataset and decode to verify the [CLS][SEP][SEP] structure.

---

## Part 4: Tokenizing the Entire Dataset

```python
tokenized_dataset = tokenizer(                                    # 55
    raw_datasets["train"]["sentence1"][:],                        # 56
    raw_datasets["train"]["sentence2"][:],                        # 57
    padding=True,                                                 # 58
    truncation=True,                                              # 59
)                                                                 # 60
```
> **Line 55-60:** Tokenize ALL pairs at once. This works but loads everything into memory simultaneously.

```python
def tokenize_function(examples):                                  # 69
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)  # 70
                                                                  #
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 72
print(f"Tokenized datasets: {tokenized_datasets}")                # 74
```
> **Line 69-74:** The **better approach** using `dataset.map()`:
> ```
> ┌─── dataset.map() — Why it's better ───────────────────────────┐
> │                                                                │
> │  1. Memory efficient — processes in chunks, not all at once   │
> │  2. Caches to disk — re-running is instant (already computed) │
> │  3. batched=True — processes 1000 examples per call (fast)    │
> │  4. Preserves all existing columns (keeps labels!)             │
> │                                                                │
> │  Before map():  {sentence1, sentence2, label, idx}             │
> │  After map():   {sentence1, sentence2, label, idx,             │
> │                  input_ids, attention_mask, token_type_ids}    │
> │                                                                │
> │  Note: no padding=True here — DataCollator pads per batch.    │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Part 5: DataCollator (Dynamic Padding)

```python
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)      # 76
samples = tokenized_datasets["train"][:8]                         # 77
print(f"Samples before tokenization: {raw_datasets['train'][:8]}")  # 78
print(f"Samples before data collator: {samples}")                 # 79
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}  # 80
print([len(x) for x in samples["input_ids"]])                     # 81
```
> **Line 76-81:** Create the collator and inspect 8 samples — each has a different `input_ids` length.
> Line 80 removes string columns (`sentence1`, `sentence2`, `idx`) that can't be converted to tensors.

```python
batch = data_collator(samples)                                    # 83
print({k: v.shape for k, v in batch.items()})                    # 84
```
> **Line 83-84:** Run the collator — it pads all samples to the same length and returns tensors:
> ```
> ┌─── Static vs Dynamic Padding ──────────────────────────────────┐
> │                                                                │
> │  Static (bad):  pad everything to 512                          │
> │  [tok tok tok 0 0 0 0 0 0 … 0 0 0]  ← 512 tokens, mostly 0s  │
> │                                                                │
> │  Dynamic (good):  pad each batch to ITS longest sample         │
> │  Batch 1 longest = 45 → all in batch padded to 45             │
> │  Batch 2 longest = 32 → all in batch padded to 32             │
> │  Batch 3 longest = 67 → all in batch padded to 67             │
> │                                                                │
> │  Output shapes for 8 samples (say longest=55):                 │
> │  {                                                             │
> │    'input_ids':      [8, 55],                                  │
> │    'attention_mask': [8, 55],                                  │
> │    'token_type_ids': [8, 55],                                  │
> │    'labels':         [8],      ← one label per sample          │
> │  }                                                             │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Key Concepts

| Concept | Purpose |
|---------|---------|
| **Labels** | Correct answers — required for supervised training |
| **Loss** | How wrong the model is (lower = better) |
| **backward()** | Compute gradients — which direction each weight should move |
| **optimizer.step()** | Actually move the weights using computed gradients |
| **dataset.map()** | Efficiently tokenize entire dataset with caching |
| **DataCollator** | Dynamic padding per batch — much more efficient than padding to 512 |
| **Sentence pairs** | BERT uses `[SEP]` + `token_type_ids` to separate two sentences |
