# put_it_all_together.py — Annotated Walkthrough

> **What this script does:** Exhaustively demos every padding and truncation option, then runs a full end-to-end sentiment prediction.

```
┌─────────────────── PADDING OPTIONS ─────────────────────────────┐
│                                                                   │
│  "longest"              → pad to longest in this batch           │
│  "max_length"           → pad to model max (512)                 │
│  "max_length"+max_length=8 → pad to custom length               │
│                                                                   │
│  Example: sentence A = 5 tokens, sentence B = 3 tokens           │
│                                                                   │
│  "longest":   A:[t t t t t]   B:[t t t 0 0]   ← both 5          │
│  "max_length":A:[t t t t t 0 0 … 0]  B:[t t t 0 0 … 0] ← 512   │
│  max_length=8:A:[t t t t t 0 0 0]  B:[t t t 0 0 0 0 0] ← 8     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
import torch                                                      # 2
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 3
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # 5
```
> **Line 1-5:** Imports and mirror setup.

```python
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"   # 7
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 8
```
> **Line 7-8:** Load the DistilBERT sentiment tokenizer.

---

## Single Sentence

```python
sequence = "I've been waiting for a HuggingFace course my whole life."  # 10
model_inputs = tokenizer(sequence)                                # 11
print(f"model_inputs: {model_inputs}")                            # 13
```
> **Line 10-13:** Minimal call — tokenize one sentence. No padding, no tensors, just Python dicts with lists.

---

## Multiple Sentences (No Padding)

```python
multiple_sequences = [                                            # 15
    "I've been waiting for a HuggingFace course my whole life.",  # 16
    "So have I!"                                                  # 17
]                                                                 # 18
model_inputs = tokenizer(multiple_sequences)                      # 19
print(f"model_inputs: {model_inputs}")                            # 21
```
> **Line 15-21:** Tokenize two sentences without padding. Output is two lists of different lengths — cannot be stacked into a single tensor yet.

---

## Padding Options

```python
model_inputs = tokenizer(multiple_sequences, padding="longest")  # 24
print(f"model_inputs, longest padding: {model_inputs}")           # 25
```
> **Line 24-25:** `padding="longest"` — pad to the longest sentence in THIS batch:
> ```
> Sentence 1: 16 tokens  ← longest
> Sentence 2:  6 tokens  → padded to 16 with [PAD]=0
>
> ✓ Efficient: minimal wasted zeros
> ✓ Best for training (each batch pads independently)
> ```

```python
model_inputs = tokenizer(multiple_sequences, padding="max_length")  # 29
print(f"model_inputs, max length padding: {model_inputs}")        # 30
```
> **Line 29-30:** `padding="max_length"` — pad ALL sentences to the model's absolute max (512 for DistilBERT):
> ```
> Sentence 1: 16 tokens → padded to 512  (496 zeros!)
> Sentence 2:  6 tokens → padded to 512  (506 zeros!)
>
> ⚠️  Very wasteful — mostly zeros
> Use only when fixed-size output is required (e.g. deployment)
> ```

```python
model_inputs = tokenizer(multiple_sequences, padding="max_length", max_length=8)  # 33
print(f"model_inputs, specified max length padding: {model_inputs}")  # 34
```
> **Line 33-34:** Custom cap — pad to exactly 8 tokens:
> ```
> Sentence 1: if longer than 8 → truncated to 8 (needs truncation=True)
> Sentence 2: if shorter than 8 → padded to 8 with zeros
>
> All outputs exactly 8 tokens wide.
> ```

---

## Truncation Options

```python
model_inputs = tokenizer(multiple_sequences, truncation=True)     # 38
print(f"model_inputs, truncation: {model_inputs}")                # 39
```
> **Line 38-39:** `truncation=True` — cut anything over 512. Short sentences are untouched.

```python
model_inputs = tokenizer(multiple_sequences, max_length=4, truncation=True)  # 42
print(f"model_inputs, specified max length truncation: {model_inputs}")  # 43
```
> **Line 42-43:** Aggressively truncate to just 4 tokens:
> ```
> "I've been waiting for a HuggingFace course my whole life."
>  → [CLS] i've been [SEP]   ← only 4 tokens, rest discarded
>
> ⚠️  Most of the sentence meaning is lost. Use with care.
> ```

---

## Return Format Options

```python
model_inputs = tokenizer(multiple_sequences, padding=True, return_tensors="pt")  # 46
print(f"model_inputs, PyTorch tensors: {model_inputs}")           # 47
```
> **Line 46-47:** Return PyTorch tensors — required to feed into a PyTorch model.

```python
model_inputs = tokenizer(multiple_sequences, padding=True, return_tensors="np")  # 50
print(f"model_inputs, NumPy arrays: {model_inputs}")              # 51
```
> **Line 50-51:** Return NumPy arrays — useful for analysis or non-PyTorch workflows.
> ```
> return_tensors options:
>   "pt" → PyTorch tensors  (use with PyTorch models)
>   "tf" → TensorFlow tensors
>   "np" → NumPy arrays
>   None → plain Python lists (default, can't feed to model directly)
> ```

---

## Special Tokens: tokenizer() vs tokenize()

```python
sequence = "I've been waiting for a HuggingFace course my whole life."  # 53
model_inputs = tokenizer(sequence)                                # 55
print(tokenizer.decode(model_inputs["input_ids"]))                # 56
```
> **Line 53-56:** Full call → includes [CLS] and [SEP] in decoded output.

```python
tokens = tokenizer.tokenize(sequence)                             # 58
ids = tokenizer.convert_tokens_to_ids(tokens)                     # 59
print(tokenizer.decode(ids))                                      # 60
```
> **Line 58-60:** Manual call → NO [CLS]/[SEP]:
> ```
> tokenizer(text):                → "[CLS] i've been … life. [SEP]"
> tokenize() + convert_to_ids():  → "i've been … life."
>                                      ↑ missing special tokens
> ```

---

## Full End-to-End Prediction

```python
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # 62
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]  # 63
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")  # 64
output = model(**tokens)                                          # 65
predictions = torch.nn.functional.softmax(output.logits, dim=-1)  # 66
print(predictions)                                                # 67
```
> **Line 62-67:** The complete pipeline done manually:
> ```mermaid
> flowchart LR
>     A["Raw text\n2 sentences"] --> B["tokenizer()\npadding+truncation\nreturn_tensors=pt"]
>     B --> C["model(**tokens)\nforward pass"]
>     C --> D["output.logits\n[2, 2]"]
>     D --> E["softmax()\n→ probabilities"]
>     E --> F["[[0.04, 0.96],\n [0.03, 0.97]]\nboth POSITIVE"]
> ```
> `**tokens` unpacks the dict — passes `input_ids`, `attention_mask` as named arguments to the model.

---

## Quick Reference

| Goal | Use |
|------|-----|
| Efficient training batches | `padding=True` (pads to longest in batch) |
| Fixed-size deployment | `padding="max_length", truncation=True, max_length=128` |
| Just inspect output | no `return_tensors` (Python lists) |
| Feed to model | `return_tensors="pt"` |
