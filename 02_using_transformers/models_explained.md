# models.py — Annotated Walkthrough

> **What this script does:** Explores loading pre-trained models, tokenizing with different options (single/batch, padding, truncation, max_length), and passing data through the model.

```
┌─────────────────── WHAT THIS SCRIPT COVERS ─────────────────────┐
│                                                                   │
│  1. Load a pre-trained BertModel                                  │
│  2. Tokenize a single sentence                                    │
│  3. Tokenize multiple sentences (batching + padding)              │
│  4. Handle very long text (truncation)                            │
│  5. Combine padding + truncation + custom max_length              │
│  6. Pass tokenized input through the model                        │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertModel  # 3
from huggingface_hub import notebook_login                        # 4
import torch                                                      # 5
```
> **Line 1-5:** Imports. `BertModel` is explicitly BERT (vs `AutoModel` which auto-detects).

---

## Loading a Pre-Trained Model

```python
# model = AutoModel.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")              # 8
```
> **Line 8:** Load BERT (~110M parameters, ~440MB download). The commented-out line above is equivalent — `AutoModel` would detect it's a BERT checkpoint and load `BertModel` anyway.
> ```
> ┌─── What are "pre-trained" weights? ────────────────────────────┐
> │                                                                │
> │  Google trained BERT on:                                       │
> │  • All of English Wikipedia (~2.5B words)                      │
> │  • BookCorpus (~800M words)                                    │
> │                                                                │
> │  Tasks it was trained on:                                      │
> │  1. Masked LM: "The [MASK] sat on the mat" → predict "cat"    │
> │  2. Next Sentence Prediction: Does B follow A?                 │
> │                                                                │
> │  from_pretrained() downloads these learned weights (~440MB)    │
> │  and loads them into memory — ready to use immediately.        │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
# model.save_pretrained('./bert-base-cased')                      # 11
```
> **Line 11 (commented out):** Would save weights locally so future runs skip the download.

---

## Tokenizing a Single Sentence

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")     # 13
```
> **Line 13:** Load BERT's tokenizer.

```python
encoded_input = tokenizer("Hello, I'm a single sentence!")       # 15
decoded_str = tokenizer.decode(encoded_input['input_ids'])        # 16
print(decoded_str)                                                # 17
```
> **Line 15-17:** Encode → decode roundtrip for a single sentence. Verifies tokenizer works correctly.

---

## Tokenizing Multiple Sentences (Batching)

```python
encoded_input = tokenizer(["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt")  # 19
print(encoded_input)                                              # 20
```
> **Line 19-20:** Tokenize two sentences together as a batch:
> ```
> ┌─── Batching with Padding ──────────────────────────────────────┐
> │                                                                │
> │  "How are you?"         → 5 tokens                             │
> │  "I'm fine, thank you!" → 8 tokens  ← the longer one          │
> │                                                                │
> │  Problem: tensors must be same shape!                          │
> │                                                                │
> │  padding=True pads shorter sentence with [PAD]=0:             │
> │  S1: [101, 1731, 1132, 1128, 102,  0,   0,   0 ]  ← padded   │
> │  S2: [101, 146,  112,  182,  1200, 1846, 999, 102]  ← full    │
> │                                                                │
> │  attention_mask: 1=real, 0=padding (model ignores padding)    │
> │  S1: [1, 1, 1, 1, 1, 0, 0, 0]                                 │
> │  S2: [1, 1, 1, 1, 1, 1, 1, 1]                                 │
> │                                                                │
> │  return_tensors="pt" → output is PyTorch tensors               │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Handling Long Text (Truncation)

```python
encoded_input = tokenizer(                                        # 22
    "This is a very very very ... very long sentence.",           # 23
    truncation=True,                                              # 24
)                                                                 # 25
print(encoded_input["input_ids"])                                 # 26
```
> **Line 22-26:** `truncation=True` chops anything longer than the model's max (512 for BERT):
> ```
> ┌─── Truncation ────────────────────────────────────────────────┐
> │                                                               │
> │  Without truncation: ERROR if input > 512 tokens              │
> │  With truncation=True: keeps first 510 tokens + [CLS] + [SEP] │
> │                                                               │
> │  [CLS] tok1 tok2 ... tok510 [SEP]  ← exactly 512              │
> │                      ↑                                        │
> │              rest of sentence discarded                        │
> └───────────────────────────────────────────────────────────────┘
> ```

---

## Padding + Truncation + Custom max_length

```python
encoded_input = tokenizer(                                        # 28
    ["How are you?", "I'm fine, thank you!"],                     # 29
    padding=True,                                                 # 30
    truncation=True,                                              # 31
    max_length=7,                                                 # 32
    return_tensors="pt",                                          # 33
)                                                                 # 34
print(encoded_input)                                              # 35
```
> **Line 28-35:** All options combined:
> ```
> ┌─── max_length=7 in action ─────────────────────────────────────┐
> │                                                                │
> │  truncation=True + max_length=7:                               │
> │    Any sentence longer than 7 tokens gets cut to 7            │
> │                                                                │
> │  padding=True:                                                 │
> │    Pad shorter sentences up to the longest (≤ max_length=7)   │
> │                                                                │
> │  Result: ALL outputs are exactly 7 tokens                      │
> │  S1: [101, tok, tok, tok, tok, 102,  0 ]  ← padded to 7       │
> │  S2: [101, tok, tok, tok, tok, tok, 102]  ← truncated to 7    │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Passing Through the Model

```python
sequences = [                                                     # 37
    "I've been waiting for a HuggingFace course my whole life.",  # 38
    "I hate this so much!",                                       # 39
]                                                                 # 40
encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")  # 41
model_inputs = torch.tensor(encoded_sequences.input_ids)          # 42
print(model_inputs)                                               # 43
```
> **Line 37-43:** Tokenize, then wrap `input_ids` in an explicit tensor.
> Note: `tokenizer(..., return_tensors="pt")` already returns a tensor, so line 42 is redundant — but shows the concept explicitly.

```python
output = model(model_inputs)                                      # 45
print(output)                                                     # 46
```
> **Line 45-46:** Pass tokens through BERT — each token gets a 768-dim context vector:
> ```mermaid
> flowchart LR
>     A["input_ids\n[2, seq_len]"] --> B["BertModel\n12 transformer layers"]
>     B --> C["last_hidden_state\n[2, seq_len, 768]"]
>     B --> D["pooler_output\n[2, 768]"]
> ```
> ```
> Output shape: [2, seq_len, 768]
>                │    │         └─ 768 dims (context-aware meaning per token)
>                │    └─ number of tokens in this sentence
>                └─ 2 sentences in the batch
>
> "bank" in "river bank"   → 768-dim vector pointing toward nature
> "bank" in "bank account" → 768-dim vector pointing toward finance
> SAME word, DIFFERENT vector because context changed — this is the magic of attention.
> ```

---

## Key Concepts

| Option | What it does |
|--------|-------------|
| `padding=True` | Pad shorter sentences so all have same length in a batch |
| `truncation=True` | Cut sentences longer than model max (512) |
| `max_length=N` | Override the max to a custom value |
| `return_tensors="pt"` | Output PyTorch tensors (required for model input) |
| Pre-trained weights | Model already learned language — you just use it |
| Hidden state | 768-dim vector capturing each token's meaning in context |
