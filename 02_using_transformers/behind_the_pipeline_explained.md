# behind_the_pipeline.py — Annotated Walkthrough

> **What this script does:** Reveals what happens inside `pipeline("sentiment-analysis")` by doing each step manually: tokenize → model → post-process.

```
┌─────────────────────── THE 3 STAGES ────────────────────────────┐
│                                                                   │
│  STAGE 1: TOKENIZATION       "I love this" → [101, 1045, ...]   │
│  STAGE 2: MODEL               [101, 1045, ...] → [-1.5, 1.6]    │
│  STAGE 3: POST-PROCESSING    [-1.5, 1.6] → "POSITIVE 96%"       │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification  # 3
import torch                                                      # 4
```
> **Line 1-4:** Imports:
>
> | Class | Role |
> |-------|------|
> | `pipeline` | Magic one-liner (all 3 stages hidden inside) |
> | `AutoTokenizer` | Text → Numbers |
> | `AutoModel` | Numbers → Hidden states (raw understanding) |
> | `AutoModelForSequenceClassification` | Numbers → Label predictions |

---

## Part 1: The Easy Way — Pipeline Does Everything

```python
classifier = pipeline("sentiment-analysis")                       # 6
```
> **Line 6:** One line creates the entire pipeline. Auto-selects `distilbert-base-uncased-finetuned-sst-2-english`.

```python
sentiment_result = classifier(                                    # 7
    [                                                             # 8
        "I've been waiting for a HuggingFace course my whole life.",  # 9
        "I hate this so much!",                                   # 10
    ]                                                             # 11
)                                                                 # 12
                                                                  #
print(sentiment_result)                                           # 14
```
> **Line 7-14:** Feed 2 sentences, print results.
> Output: `[{'label': 'POSITIVE', 'score': 0.96}, {'label': 'NEGATIVE', 'score': 0.99}]`

---

## Part 2: Stage 1 — Tokenization (Text → Numbers)

```python
checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'   # 16
```
> **Line 16:** The model's name on HuggingFace Hub, decoded:
> ```
> distilbert    → smaller/faster BERT (distilled = compressed version)
> base          → medium size
> uncased       → case-insensitive ("Hello" = "hello")
> finetuned     → additionally trained on a specific task
> sst-2-english → Stanford Sentiment Treebank (movie reviews), English
> ```

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint)             # 17
```
> **Line 17:** Load the tokenizer matching this model — contains the vocabulary (word→number dictionary).

```python
raw_inputs = [                                                    # 19
    "I've been waiting for a HuggingFace course my whole life.",  # 20
    "I hate this so much!",                                       # 21
]                                                                 # 22
```
> **Line 19-22:** Plain text — what humans read before any processing.

```python
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")  # 24
print(inputs)                                                     # 25
```
> **Line 24-25:** Converts text → model-ready numbers in one call:
> ```
> ┌─── What each argument does ───────────────────────────────────┐
> │                                                               │
> │  padding=True:                                                │
> │    Sentence 1 has 16 tokens, Sentence 2 has 9.               │
> │    GPUs need equal-length batches, so pad shorter one:        │
> │                                                               │
> │    S1: [tok tok tok tok tok tok tok tok tok tok tok tok tok tok tok tok]
> │    S2: [tok tok tok tok tok tok tok tok tok  0   0   0   0   0   0   0]
> │                                              └── padding (zeros)
> │                                                               │
> │    attention_mask marks which are real (1) vs padding (0):   │
> │    S1: [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]                    │
> │    S2: [1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0]                    │
> │                                                               │
> │  truncation=True:  cut anything longer than 512 tokens        │
> │  return_tensors="pt": return PyTorch tensors (not Python lists)│
> └───────────────────────────────────────────────────────────────┘
> ```

---

## Part 3: Stage 2a — Base Model (Raw Hidden States)

```python
model = AutoModel.from_pretrained(checkpoint)                     # 27
outputs = model(**inputs)                                         # 28
```
> **Line 27-28:** Load the base model and run a **forward pass** (data flows through the network):
>
> ```mermaid
> flowchart LR
>     A["input_ids\n[2, 16]"] --> B["Embedding\nID → 768-dim vector"]
>     B --> C["6× Transformer\nLayer attention+FFN"]
>     C --> D["Hidden states\n[2, 16, 768]"]
> ```
>
> `**inputs` unpacks the dict — equivalent to writing `model(input_ids=..., attention_mask=...)`.

```python
first_token_vector = outputs.last_hidden_state[0, 0, :]          # 29
```
> **Line 29:** Extract one specific vector — first sentence, first token, all 768 dimensions:
> ```
> outputs.last_hidden_state  shape: [2,  16, 768]
>                                    │    │    └─ 768 values (meaning of this token)
>                                    │    └─ 16 token positions
>                                    └─ 2 sentences in batch
>
> [0, 0, :]  →  first sentence, first token ([CLS]), all dimensions
> ```

```python
print('first sentence and first token and all hidden vector: ', first_token_vector)  # 30
vector_as_list = first_token_vector.tolist()                      # 31
print('first sentence and first token and all hidden vector as list: ', vector_as_list)  # 32
```
> **Line 30-32:** Shows 768 floats — the model's internal contextual "understanding" of the [CLS] token.
> `.tolist()` just converts from PyTorch tensor to a plain Python list for easier printing.

---

## Part 4: Stage 2b — Classification Model (Predictions)

```python
sequence_classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint)  # 34
```
> **Line 34:** Same transformer layers, but WITH a classification head added on top:
> ```
> ┌─── Architecture Comparison ─────────────────────────────────┐
> │                                                              │
> │  AutoModel:                                                  │
> │    [Transformer × 6] ──→ hidden states [2, 16, 768]          │
> │                          "I understand the text"             │
> │                                                              │
> │  AutoModelForSequenceClassification:                         │
> │    [Transformer × 6] ──→ [Linear: 768 → 2] ──→ logits [2, 2]│
> │                           ↑ classification head              │
> │                          "I can label the text"              │
> └──────────────────────────────────────────────────────────────┘
> ```

```python
classification_outputs = sequence_classifier(**inputs)             # 35
print('classification outputs: ', classification_outputs)         # 36
```
> **Line 35-36:** Output has `.logits` — raw scores like `[[-1.5, 1.6], [2.3, -1.8]]`. Not probabilities yet.

---

## Part 5: Stage 3 — Post-Processing (Logits → Probabilities)

```python
predictions = torch.nn.functional.softmax(classification_outputs.logits, dim=-1)  # 37
print('predictions: ', predictions)                               # 38
```
> **Line 37-38:** **Softmax** squishes any numbers into [0,1] probabilities that sum to 1:
> ```
> ┌─── Softmax Visualization ──────────────────────────────────────┐
> │                                                                │
> │  Logits:        [-1.5,  1.6]   (any real numbers)              │
> │                    ↓ softmax                                   │
> │  Probabilities: [0.04, 0.96]   (positive, sum to 1.0)          │
> │                  NEG   POS                                     │
> │                                                                │
> │  Formula:  e^x_i / Σ e^x_j                                    │
> │  e^(-1.5)=0.22, e^(1.6)=4.95, total=5.17                     │
> │  → [0.22/5.17, 4.95/5.17] = [0.04, 0.96]                     │
> │                                                                │
> │  dim=-1: apply softmax along the last dimension (the 2 scores) │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
print(sequence_classifier.config.id2label)                        # 40
```
> **Line 40:** Prints `{0: 'NEGATIVE', 1: 'POSITIVE'}` — index 0 = negative, index 1 = positive.

---

## Key Concepts

| Concept | Analogy |
|---------|---------|
| **Padding** | Adding blank pages so all books are the same thickness |
| **Hidden states** | Model's "thoughts" about each word — 768-dim vector |
| **Classification head** | A "decision layer" bolted on top of the base model |
| **Logits** | Raw unnormalized scores |
| **Softmax** | Converts raw scores → percentages that sum to 100% |
| **Forward pass** | Running data through the network — no learning, just computation |
