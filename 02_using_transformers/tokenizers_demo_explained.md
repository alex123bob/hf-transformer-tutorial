# tokenizers_demo.py — Annotated Walkthrough

> **What this script does:** Demonstrates tokenization step by step — splitting text into subwords, converting to IDs, and decoding back to text.

```
┌─────────────────── TOKENIZATION PIPELINE ───────────────────────┐
│                                                                   │
│  "Using a Transformer network is simple"                          │
│         ↓  tokenize()                                             │
│  ['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']  │
│         ↓  convert_tokens_to_ids()                                │
│  [7993, 170, 13809, 21877, 2897, 1110, 3014]                     │
│                                                                   │
│  Calling tokenizer() directly does ALL above + adds:             │
│  [101, 7993, 170, 13809, 21877, 2897, 1110, 3014, 102]           │
│   [CLS]                                           [SEP]           │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
from transformers import BertTokenizer                            # 3
```
> **Line 1-3:** Setup. `BertTokenizer` is BERT's specific tokenizer (you could also use `AutoTokenizer`).

```python
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")     # 5
```
> **Line 5:** Load BERT's vocabulary of ~28,996 tokens. `cased` = case-sensitive ("Hello" ≠ "hello").
> ```
> ┌─── Inside the tokenizer ───────────────────────────────────────┐
> │                                                                │
> │  Vocabulary (lookup table):                                    │
> │  ┌──────────────┬────────┐                                     │
> │  │  Token       │  ID    │                                     │
> │  ├──────────────┼────────┤                                     │
> │  │  [PAD]       │  0     │ ← filler for padding                │
> │  │  [CLS]       │  101   │ ← start-of-sequence                 │
> │  │  [SEP]       │  102   │ ← end / separator                   │
> │  │  Using       │  7993  │                                     │
> │  │  Trans       │  13809 │                                     │
> │  │  ##former    │  21877 │ ← ## = continues previous word      │
> │  └──────────────┴────────┘                                     │
> └────────────────────────────────────────────────────────────────┘
> ```

---

## Full Tokenization (All-in-One)

```python
encoded_input = tokenizer("Using a Transformer network is simple")  # 6
print(encoded_input)                                              # 7
```
> **Line 6-7:** Full pipeline in one call: split → IDs → add special tokens:
> ```python
> # Output:
> {
>     'input_ids':      [101, 7993, 170, 13809, 21877, 2897, 1110, 3014, 102],
>     #                  [CLS] Using   a   Trans ##form network  is simple [SEP]
>     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],  # single sentence → all 0
>     'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1],  # all real tokens → all 1
> }
> ```

---

## Breaking It Into Steps

```python
tokens = tokenizer.tokenize("Using a Transformer network is simple")  # 9
print(tokens)                                                     # 10
```
> **Line 9-10:** **Step 1 only** — split into tokens, no IDs, no special tokens:
> ```
> ┌─── WordPiece Splitting ────────────────────────────────────────┐
> │                                                                │
> │  "Using"       in vocab? YES → ['Using']                       │
> │  "a"           in vocab? YES → ['a']                           │
> │  "Transformer" in vocab? NO                                    │
> │      "Trans"   in vocab? YES → keep 'Trans'                    │
> │      "former"  in vocab? YES → keep '##former'  (## = suffix)  │
> │  "network"     in vocab? YES → ['network']                     │
> │  "is"          in vocab? YES → ['is']                          │
> │  "simple"      in vocab? YES → ['simple']                      │
> │                                                                │
> │  Result: ['Using', 'a', 'Trans', '##former',                  │
> │           'network', 'is', 'simple']                           │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
input_ids = tokenizer.convert_tokens_to_ids(tokens)              # 11
print(input_ids)                                                  # 12
```
> **Line 11-12:** **Step 2 only** — dictionary lookup, token → ID:
> ```
> 'Using'    → 7993
> 'a'        → 170
> 'Trans'    → 13809
> '##former' → 21877   ...etc
>
> Result: [7993, 170, 13809, 21877, 2897, 1110, 3014]
>           ↑ no 101 ([CLS]) or 102 ([SEP]) — those only appear in the full call
> ```

---

## Decoding (Numbers → Text)

```python
decoded_output = tokenizer.decode(encoded_input.input_ids)        # 14
print(decoded_output)                                             # 15
```
> **Line 14-15:** Reverse the process — IDs back to human-readable text:
> ```
> ┌─── Decode Process ────────────────────────────────────────────┐
> │                                                               │
> │  [101, 7993, 170, 13809, 21877, 2897, 1110, 3014, 102]       │
> │   ↓      ↓    ↓     ↓      ↓      ↓     ↓     ↓    ↓        │
> │  [CLS] Using  a   Trans ##former network  is  simple [SEP]   │
> │                    └──────┬──────┘                            │
> │                      merged → "Transformer"  (## removed)     │
> │                                                               │
> │  Output: "[CLS] Using a Transformer network is simple [SEP]"  │
> └───────────────────────────────────────────────────────────────┘
> ```

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Why tokenize?** | Models only understand numbers, not words |
| **WordPiece** | Splits unknown words into known subwords — no word is ever truly "unknown" |
| **## prefix** | Marks a subword continuation: `Trans` + `##former` = `Transformer` |
| **[CLS]** | Start token — its hidden state often summarises the whole sentence for classification |
| **[SEP]** | End/separator token — marks sentence boundaries |

```
┌─── WHY SUBWORDS? ─────────────────────────────────────────────┐
│                                                                │
│  "unhappiness"  → ["un", "##happiness"]                        │
│  "ChatGPT"      → ["Chat", "##G", "##PT"]                     │
│  "supercalifragilisticexpialidocious"  → many small pieces     │
│                                                                │
│  Benefits:                                                      │
│  ✓ ~30K tokens can represent ANY English text                  │
│  ✓ No "unknown word" problem                                   │
│  ✓ Related words share pieces → shared understanding           │
└────────────────────────────────────────────────────────────────┘
```
