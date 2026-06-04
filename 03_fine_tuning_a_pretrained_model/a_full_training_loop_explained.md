# a_full_training_loop.py — Annotated Walkthrough

> **What this script does:** Implements the entire training loop manually — no Trainer wrapper. Every single step is explicit. This is the most important script for truly understanding how model training works.

```
┌─────────────────── THE TRAINING LOOP ───────────────────────────┐
│                                                                   │
│  This is what all ML training fundamentally is:                  │
│                                                                   │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐      │
│  │ Get Batch│ → │ Forward  │ → │ Backward │ → │ Update   │      │
│  │          │   │  Pass    │   │  Pass    │   │ Weights  │      │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘      │
│  "next 8        "predict,       "which way      "take a step     │
│   samples"       compute loss"   to improve?"    that direction" │
│                                                                   │
│  Repeat ~1374 times (3 epochs × 458 batches)                     │
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, get_scheduler  # 8
from torch.utils.data import DataLoader                           # 9
from datasets import load_dataset                                 # 11
from tqdm.auto import tqdm                                        # 12
```
> **Line 1-12:** New imports vs previous scripts:
>
> | Import | Role |
> |--------|------|
> | `DataLoader` | PyTorch's batch iterator — feeds data to the model in chunks |
> | `get_scheduler` | Creates a learning rate schedule (gradually reduces lr) |
> | `tqdm` | Progress bar — shows estimated time and step count |

---

## Data Preparation

```python
raw_datasets = load_dataset("glue", "mrpc")                       # 15
checkpoint = "bert-base-uncased"                                  # 16
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 17
                                                                  #
def tokenize_function(examples):                                  # 19
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)  # 20
                                                                  #
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)  # 22
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)      # 23
print(f"tokenized datasets structure: {tokenized_datasets}")      # 24
```
> **Line 15-24:** Same data prep as before — load MRPC, tokenize all sentence pairs.

---

## Formatting Dataset for PyTorch

```python
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])  # 26
```
> **Line 26:** Remove text columns the model can't use — strings can't be converted to tensors:
> ```
> Before: {sentence1, sentence2, idx, label, input_ids, attention_mask, token_type_ids}
> After:  {label, input_ids, attention_mask, token_type_ids}
> ```

```python
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # 27
```
> **Line 27:** The model expects the key `"labels"` (plural), but the dataset column is `"label"` (singular). Rename to match.

```python
tokenized_datasets.set_format("torch")                            # 28
print(f"tokenized datasets structure after processing: {tokenized_datasets.column_names}")  # 30
```
> **Line 28-30:** Tell the dataset to return PyTorch tensors automatically when indexed:
> ```
> Before set_format:  sample["input_ids"] → [101, 1045, ...]   (Python list)
> After  set_format:  sample["input_ids"] → tensor([101, 1045, ...])  (PyTorch)
> ```

---

## DataLoader (Batch Iterator)

```python
print(f"construct data loaders")                                  # 32
train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator)  # 33
eval_dataloader = DataLoader(tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator)  # 34
```
> **Line 32-34:** Create DataLoaders — objects that feed data in batches:
> ```
> ┌─── What DataLoader does ───────────────────────────────────────┐
> │                                                                │
> │  Dataset has 3668 samples. DataLoader groups them:             │
> │                                                                │
> │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  batch 1 (samples 1-8)            │
> │  └──┴──┴──┴──┴──┴──┴──┴──┘                                    │
> │  ┌──┬──┬──┬──┬──┬──┬──┬──┐  batch 2 (samples 9-16)           │
> │  └──┴──┴──┴──┴──┴──┴──┴──┘                                    │
> │  ...                          458 batches total                │
> │                                                                │
> │  shuffle=True   → randomise order each epoch                   │
> │                    (stops model memorising the sequence)       │
> │  batch_size=8   → 8 samples per batch                          │
> │  collate_fn     → dynamic padding applied to each batch        │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Model and Quick Sanity Check

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)  # 36
```
> **Line 36:** Load BERT with a 2-class classification head.

```python
for batch in train_dataloader:                                    # 38
    print({k: v for k, v in batch.items()})                      # 39
    outputs = model(**batch)                                      # 40
    print(outputs.loss, outputs.logits.shape)                     # 41
    break                                                         # 42
```
> **Line 38-42:** Grab ONE batch, run it through the model, print loss + output shape, then stop:
> ```
> ┌─── Sanity Check ───────────────────────────────────────────────┐
> │                                                                │
> │  batch shapes:                                                 │
> │    input_ids:      [8, max_len]                                │
> │    attention_mask: [8, max_len]                                │
> │    token_type_ids: [8, max_len]                                │
> │    labels:         [8]                                         │
> │                                                                │
> │  outputs.loss: ~0.69  (random model ≈ -ln(0.5) for 2 classes) │
> │  outputs.logits.shape: [8, 2]                                  │
> │                                                                │
> │  If no error → data pipeline works correctly. ✓                │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Optimizer and Learning Rate Scheduler

```python
optimizer = AdamW(model.parameters(), lr=5e-5)                    # 44
```
> **Line 44:** AdamW optimizer with learning rate `5e-5 = 0.00005`. Standard for fine-tuning transformers.
> ```
> Too big (0.01):   loss oscillates, model may diverge
> Just right (5e-5):smooth descent → best weights
> Too small (1e-8): takes forever, barely learns
> ```

```python
num_epochs = 3                                                    # 46
num_training_steps = num_epochs * len(train_dataloader)           # 47
lr_scheduler = get_scheduler(                                     # 48
    "linear",                                                     # 49
    optimizer=optimizer,                                          # 50
    num_warmup_steps=0,                                           # 51
    num_training_steps=num_training_steps,                        # 52
)                                                                 # 53
print(num_training_steps)                                         # 54
```
> **Line 46-54:** Linear learning rate schedule — gradually reduce lr to zero over training:
> ```
> ┌─── Learning Rate Schedule ─────────────────────────────────────┐
> │                                                                │
> │  num_training_steps = 3 × 458 = 1374 steps                    │
> │                                                                │
> │  lr                                                            │
> │  5e-5 ┤╲                                                       │
> │       │  ╲                                                     │
> │       │    ╲                                                   │
> │       │      ╲                                                 │
> │  0    ┤────────╲                                               │
> │       └──────────→ step 1374                                   │
> │                                                                │
> │  Why? Start fast (big lr), refine slowly (small lr).           │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Device Setup

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # 56
model.to(device)                                                  # 57
print(device)                                                     # 58
```
> **Line 56-58:** Move model to GPU if available. Input batches must also be moved to the same device.

---

## ⭐ THE TRAINING LOOP

```python
progress_bar = tqdm(range(num_training_steps))                    # 60
```
> **Line 60:** Progress bar — shows `[████████░░░░] 60% | 824/1374 | loss: 0.31 | 2:14<1:28`.

```python
model.train()                                                     # 62
```
> **Line 62:** Switch model to training mode — enables dropout (randomly ignores ~10% of neurons each step to prevent over-reliance on any single one).

```python
for epoch in range(num_epochs):                                   # 63
    for batch in train_dataloader:                                # 64
```
> **Line 63-64:** Outer loop = epochs (3 full passes). Inner loop = batches (458 per epoch).

```python
        batch = {k: v.to(device) for k, v in batch.items()}      # 65
```
> **Line 65:** Move every tensor in the batch to the same device as the model. If the model is on GPU and batch is on CPU → error.

```python
        outputs = model(**batch)                                  # 66
        loss = outputs.loss                                       # 67
```
> **Line 66-67:** **Forward pass** — model predicts all 8 samples and computes the average loss for this batch.

```python
        loss.backward()                                           # 68
```
> **Line 68:** **Backward pass** — compute gradient for every one of the 110M parameters:
> ```
> ┌─── What backward() does ───────────────────────────────────────┐
> │                                                                │
> │  PyTorch tracked every operation in forward().                 │
> │  backward() replays them in reverse (chain rule) to compute:  │
> │                                                                │
> │  ∂loss/∂weight  for every weight in the network               │
> │  (= "if I increase this weight, does loss go up or down?")     │
> │                                                                │
> │  Result: each parameter now has a .grad attribute pointing    │
> │  in the direction that would INCREASE loss.                    │
> │  The optimizer will step in the OPPOSITE direction.            │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
        optimizer.step()                                          # 70
```
> **Line 70:** **Update weights** — for each parameter: `weight -= lr × gradient`

```python
        lr_scheduler.step()                                       # 71
```
> **Line 71:** **Decay the learning rate** — slightly reduce lr according to the linear schedule.

```python
        optimizer.zero_grad()                                     # 72
```
> **Line 72:** **Reset gradients to zero** — CRITICAL step, easy to forget:
> ```
> ┌─── Why zero_grad() is critical ────────────────────────────────┐
> │                                                                │
> │  PyTorch ACCUMULATES gradients by default.                     │
> │  Without zero_grad():                                          │
> │                                                                │
> │  Step 1: backward() → grad = 0.5                               │
> │  Step 2: backward() → grad = 0.5 + 0.4 = 0.9  ← WRONG!       │
> │  Step 3: backward() → grad = 0.9 + 0.3 = 1.2  ← WORSE!       │
> │                                                                │
> │  With zero_grad() before each backward():                      │
> │  Step 1: zero → 0,  backward → 0.5  ✓                         │
> │  Step 2: zero → 0,  backward → 0.4  ✓                         │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
        progress_bar.update(1)                                    # 73
```
> **Line 73:** Advance the progress bar by 1 step.

---

## The Evaluation Loop

```python
metric = evaluate.load('glue', 'mrpc')                            # 75
model.eval()                                                      # 76
```
> **Line 75-76:** Load MRPC metric (accuracy + F1). Switch to eval mode — disables dropout.

```python
for batch in eval_dataloader:                                     # 77
    batch = {k: v.to(device) for k, v in batch.items()}          # 78
    with torch.no_grad():                                         # 79
        outputs = model(**batch)                                  # 80
```
> **Line 77-80:** Run each eval batch through the model:
> ```
> torch.no_grad()
>   → don't build the computational graph for gradients
>   → saves memory and compute (we're not calling backward() here)
> ```

```python
    logits = outputs.logits                                       # 82
    predictions = torch.argmax(logits, dim=-1)                    # 83
    metric.add_batch(predictions=predictions, references=batch["labels"])  # 84
```
> **Line 82-84:** Convert logits → predicted class index, accumulate across all batches:
> ```
> logits:      [[-1.2, 2.3], [1.5, -0.8], …]   shape [8, 2]
> argmax:      [1, 0, …]                         shape [8]
>               ↑ pick index of highest score
>
> add_batch() collects predictions from ALL 51 eval batches
> before computing the final metric (otherwise per-batch numbers are noisy).
> ```

```python
result = metric.compute()                                         # 86
print(result)                                                     # 87
```
> **Line 86-87:** Compute final metrics across all 408 validation samples:
> Output: `{'accuracy': 0.857, 'f1': 0.896}`

---

## Full Timeline

```
┌─── What happens when you run this script ──────────────────────┐
│                                                                │
│  ① Load + tokenize dataset              ~10 seconds            │
│  ② Prepare DataLoaders                  ~1 second              │
│  ③ Load pre-trained BERT                ~5 seconds             │
│  ④ Training loop  (1374 steps)          ~5–30 minutes          │
│     [████████████████░░░░░░] 74%                               │
│     loss: 0.69 → 0.45 → 0.32 → 0.21  (steadily improving)     │
│  ⑤ Evaluation                           ~30 seconds            │
│     {'accuracy': 0.857, 'f1': 0.896}                           │
│                                                                │
│  Before training: ~50% accuracy (random guessing)              │
│  After training:  ~86% accuracy (actually learned the task!)   │
└────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

| Concept | Analogy |
|---------|---------|
| **Forward pass** | Student takes a test, writes answers |
| **Loss** | Score on the test (lower = better here) |
| **Backward pass** | Teacher marks which answers were wrong and how |
| **optimizer.step()** | Student studies the mistakes and updates their knowledge |
| **zero_grad()** | Erase the scratch paper before the next question |
| **Learning rate** | How aggressively to study — too much burns out, too little never improves |
| **LR scheduler** | Study hard early, review lightly later |
| **model.train()** | "Exam practice mode" — dropout is active, model must not rely on any one neuron |
| **model.eval()** | "Real exam mode" — use full model capacity, no dropout |
| **torch.no_grad()** | "Just check answers, don't take notes" — skip gradient tracking |
