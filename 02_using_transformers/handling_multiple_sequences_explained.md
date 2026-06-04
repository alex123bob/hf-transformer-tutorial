# handling_multiple_sequences.py — Annotated Walkthrough

> **What this script does:** Shows the difference between manual step-by-step tokenization vs the all-in-one call, and why the all-in-one approach is always preferred.

```
┌─────────────── TWO WAYS TO TOKENIZE ────────────────────────────┐
│                                                                   │
│  WAY 1 — Step by step (manual):                                  │
│    tokenize() → convert_tokens_to_ids() → torch.tensor()         │
│    ⚠️  Misses [CLS] and [SEP] special tokens!                    │
│                                                                   │
│  WAY 2 — All-in-one (correct):                                   │
│    tokenizer(text, return_tensors="pt")                          │
│    ✓ Adds special tokens, attention_mask, ready for model        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
import torch                                                      # 2
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 3
os.environ['TOKENIZERS_PARALLELISM'] = 'false'                    # 4
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 5
```
> **Line 1-5:** Setup. `TOKENIZERS_PARALLELISM=false` suppresses a warning about parallel tokenization in multi-process environments.

```python
device = torch.device("cpu")                                      # 8
```
> **Line 8:** Force CPU to avoid MPS (Apple GPU) quirks with this specific script.

```python
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"   # 11
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 12
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)  # 13
```
> **Line 11-13:** Load tokenizer and model, move model to CPU.
> ```
> ┌─── .to(device) explained ──────────────────────────────────────┐
> │                                                                │
> │  A model's ~66M parameters are just numbers stored in memory.  │
> │  .to(device) moves all those numbers to:                       │
> │    "cpu"  → system RAM    (slow but always works)              │
> │    "cuda" → GPU VRAM      (fast, NVIDIA)                       │
> │    "mps"  → Apple GPU     (fast, M-series Macs)                │
> │                                                                │
> │  ⚠️  Input tensors MUST be on the same device as the model!    │
> │     That's why we call .to(device) on batches in training too. │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Manual Tokenization (Step by Step)

```python
sequence = "I've been waiting for a HuggingFace course my whole life."  # 15
```
> **Line 15:** Our input sentence.

```python
tokens = tokenizer.tokenize(sequence)                             # 17
ids = tokenizer.convert_tokens_to_ids(tokens)                     # 18
print(f"ids: {ids}")                                              # 19
```
> **Line 17-19:** Manual two-step tokenization:
> ```
> ┌─── Step by Step ───────────────────────────────────────────────┐
> │                                                                │
> │  Step 1 — tokenize():                                          │
> │  "I've been waiting..." → ['i', "'", 've', 'been', 'waiting',  │
> │                            'for', 'a', 'hugging', '##face', …] │
> │                                                                │
> │  Step 2 — convert_tokens_to_ids():                             │
> │  ['i', "'", 've', 'been', …] → [1045, 1005, 2310, 2042, …]   │
> │                                                                │
> │  ⚠️  MISSING:  101 ([CLS]) at start, 102 ([SEP]) at end!       │
> │     The model expects these special tokens.                    │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## All-in-One Tokenization (Correct Way)

```python
tokenized_input = tokenizer(sequence, return_tensors="pt")        # 21
print(f"tokenized_input: {tokenized_input}")                      # 22
```
> **Line 21-22:** The correct way — does everything automatically:
> ```
> ┌─── Manual vs All-in-One ───────────────────────────────────────┐
> │                                                                │
> │  Manual ids:     [1045, 1005, 2310, 2042, …]                   │
> │                   ↑ no [CLS]          no [SEP] ↑               │
> │                                                                │
> │  All-in-one:     [101, 1045, 1005, 2310, 2042, …, 102]         │
> │                   ↑ [CLS]                    [SEP] ↑           │
> │                                                                │
> │  Also returns:   attention_mask: [1, 1, 1, 1, …, 1]            │
> │                  (all ones = all real tokens, no padding here)  │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Running the Model

```python
input_ids = torch.tensor([ids]).to(device)                        # 24
print("Input IDs:", input_ids)                                    # 25
```
> **Line 24-25:** Wrap the manual IDs in a batch dimension and move to device:
> ```
> ids = [1045, 1005, 2310, …]        ← 1D list, length = seq_len
> torch.tensor([ids])                 ← 2D: [1, seq_len]  (batch of 1)
>                                          ↑ models always expect a batch dimension
> ```

```python
output = model(input_ids)                                         # 27
print("Logits:", output.logits)                                   # 28
```
> **Line 27-28:** Forward pass — get logits (raw scores for NEGATIVE / POSITIVE).
> ```
> Output logits shape: [1, 2]
>                       │  └─ 2 classes (NEGATIVE, POSITIVE)
>                       └─ 1 sentence in batch
>
> ⚠️  Using the manual ids (without [CLS]/[SEP]) may give
>     slightly off results vs the proper all-in-one tokenizer call.
> ```

---

## Key Takeaway

```
┌─── ALWAYS USE THE ALL-IN-ONE CALL ─────────────────────────────┐
│                                                                 │
│  ✗ tokenizer.tokenize() + convert_tokens_to_ids()               │
│    Missing special tokens, no attention_mask, not tensor        │
│                                                                 │
│  ✓ tokenizer(text, return_tensors="pt")                         │
│    Adds [CLS]/[SEP], returns attention_mask, returns tensor     │
│    Works for single sentences, batches, pairs — everything       │
│                                                                 │
│  The manual steps are only useful for LEARNING what's inside.   │
└─────────────────────────────────────────────────────────────────┘
```
