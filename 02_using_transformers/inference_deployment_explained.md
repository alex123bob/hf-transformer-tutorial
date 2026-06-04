# inference_deployment.py — Annotated Walkthrough

> **What this script does:** Implements a local chatbot using a **generative** language model (SmolLM3-3B). Unlike classification, this model produces text token-by-token — like a mini ChatGPT running on your laptop.

```
┌─────────────── CLASSIFICATION vs GENERATION ────────────────────┐
│                                                                   │
│  Classification (previous scripts):                              │
│  Input: "I love this movie" → Output: POSITIVE  (one label)     │
│                                                                   │
│  Generation (this script):                                       │
│  Input: "What is Python?" → Output: "Python is a programming    │
│          language created by Guido van Rossum in…" (free text)   │
│                                                                   │
│  Autoregressive loop — generates one token at a time:            │
│  "What is" → "Python"                                            │
│  "What is Python" → "?"                                          │
│  "What is Python?" → "Python"                                    │
│  "What is Python? Python" → "is"    …and so on                  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
import torch                                                      # 4
from transformers import AutoTokenizer, AutoModelForCausalLM      # 5
```
> **Line 1-5:** `AutoModelForCausalLM` = "Causal Language Model" — predicts the NEXT token from all previous tokens. This is how GPT-style models work (left-to-right only).

```python
MODEL_ID = "HuggingFaceTB/SmolLM3-3B"                            # 7
```
> **Line 7:** 3 billion parameter model — small enough to run locally (~6GB in float16).

```python
if torch.backends.mps.is_available():                             # 12
    DEVICE = "mps"                                                # 13
elif torch.cuda.is_available():                                   # 14
    DEVICE = "cuda"                                               # 15
else:                                                             # 16
    DEVICE = "cpu"                                                # 17
```
> **Line 12-17:** Device priority ladder: Apple GPU → NVIDIA GPU → CPU.

---

## load_model()

```python
def load_model(model_id: str = MODEL_ID):                         # 20
    print(f"Loading model '{model_id}' on device '{DEVICE}' …")  # 21
    tokenizer = AutoTokenizer.from_pretrained(model_id)           # 22
    model = AutoModelForCausalLM.from_pretrained(                 # 23
        model_id,                                                  # 24
        dtype=torch.float16 if DEVICE != "cpu" else torch.float32,  # 25
    )                                                              # 26
    model = model.to(DEVICE)                                      # 27
    model.eval()                                                   # 28
    print("Model loaded.\n")                                       # 29
    return tokenizer, model                                        # 30
```
> **Line 20-30:**
> ```
> ┌─── dtype: float16 vs float32 ─────────────────────────────────┐
> │                                                               │
> │  float32 (default): 32 bits per weight → ~12GB for 3B model   │
> │  float16 (half):    16 bits per weight → ~6GB for 3B model    │
> │                                                               │
> │  float16 = half the memory, nearly same quality               │
> │  Only works on GPU (CPU must use float32)                     │
> │                                                               │
> │  model.eval() disables dropout so inference is deterministic. │
> └───────────────────────────────────────────────────────────────┘
> ```

---

## generate_reply()

```python
def generate_reply(                                               # 33
    tokenizer,                                                    # 34
    model,                                                        # 35
    conversation: list[dict],                                     # 36
    max_new_tokens: int = 512,                                    # 37
    temperature: float = 0.7,                                     # 38
    top_p: float = 0.9,                                           # 39
) -> str:                                                         # 40
```
> **Line 33-40:** Generation parameters:
> ```
> ┌─── temperature ────────────────────────────────────────────────┐
> │  Controls randomness / creativity:                             │
> │  0.0  → always pick most likely token  (deterministic, boring) │
> │  0.7  → some variation                 (good default)          │
> │  1.5+ → very random                    (creative but unstable) │
> └────────────────────────────────────────────────────────────────┘
>
> ┌─── top_p (nucleus sampling) ───────────────────────────────────┐
> │  Only sample from tokens whose cumulative probability ≤ top_p  │
> │  Example distribution:                                         │
> │  "mat"=50%, "couch"=20%, "floor"=15%, "rug"=5%, …             │
> │  top_p=0.9 → consider: mat+couch+floor+rug (sum=90%)          │
> │             → discard everything else (the "long tail")        │
> │  Prevents picking very unlikely/nonsensical tokens.            │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
    text = tokenizer.apply_chat_template(                         # 46
        conversation,                                             # 47
        tokenize=False,                                           # 48
        add_generation_prompt=True,                               # 49
        enable_thinking=False,                                    # 50
    )                                                             # 51
    print(f"full prompt:\n{text}\n")                              # 52
```
> **Line 46-52:** Format the conversation list into the model's expected text format:
> ```
> conversation = [
>   {"role": "system",    "content": "You are helpful…"},
>   {"role": "user",      "content": "What is Python?"},
> ]
>
> apply_chat_template() converts to:
>
>   <|system|>You are helpful…<|end|>
>   <|user|>What is Python?<|end|>
>   <|assistant|>            ← add_generation_prompt=True adds this
>                              tells the model: "your turn to reply"
>
> Each model has its own special format — apply_chat_template()
> automatically uses the correct one for this tokenizer.
> ```

```python
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)      # 53
    print(f"input_ids shape: {inputs}, {inputs['input_ids'].shape}")  # 54
```
> **Line 53-54:** Tokenize the formatted prompt and move to GPU.

```python
    with torch.no_grad():                                         # 56
        output_ids = model.generate(                              # 57
            **inputs,                                             # 58
            max_new_tokens=max_new_tokens,                        # 59
            do_sample=True,                                       # 60
            temperature=temperature,                              # 61
            top_p=top_p,                                          # 62
            pad_token_id=tokenizer.eos_token_id,                  # 63
        )                                                         # 64
```
> **Line 56-64:** Generate tokens one at a time:
> ```
> ┌─── model.generate() — Autoregressive Loop ────────────────────┐
> │                                                               │
> │  Unlike model(**inputs) which does ONE forward pass,          │
> │  model.generate() loops up to max_new_tokens times:           │
> │                                                               │
> │  Iteration 1: feed prompt → predict token "Python"            │
> │  Iteration 2: feed prompt+"Python" → predict "is"             │
> │  Iteration 3: feed prompt+"Python is" → predict "a"           │
> │  … until <|end|> token or max_new_tokens reached              │
> │                                                               │
> │  do_sample=True: randomly sample (vs greedy = always pick #1) │
> │  torch.no_grad(): skip gradient tracking — saves memory       │
> └───────────────────────────────────────────────────────────────┘
> ```

```python
    print(f"output_ids shape: {output_ids.shape}, output_ids: {output_ids}")  # 67
    print(f"input_ids length: {inputs['input_ids'].shape[1]}")    # 68
    new_token_ids = output_ids[0][inputs["input_ids"].shape[1]:]  # 69
    reply = tokenizer.decode(new_token_ids, skip_special_tokens=True)  # 70
    return reply.strip()                                           # 71
```
> **Line 67-71:** Extract only the newly generated tokens (not the prompt):
> ```
> output_ids = [ prompt_tokens… | generated_tokens… ]
>               └── we sent this ┘└── model made this ┘
>
> inputs["input_ids"].shape[1] = length of prompt
> output_ids[0][prompt_len:]   = just the reply tokens
>
> tokenizer.decode(new_token_ids, skip_special_tokens=True)
>   → removes <|end|> etc., returns clean text string
> ```

---

## chat_loop()

```python
def chat_loop(tokenizer, model):                                  # 74
    system_prompt = (                                             # 79
        "You are a helpful, concise, and honest AI assistant."    # 80
    )                                                             # 81
    conversation: list[dict] = [{"role": "system", "content": system_prompt}]  # 82
```
> **Line 74-82:** Initialize conversation with a system prompt — this shapes the model's personality.

```python
    while True:                                                   # 84
        user_input = input("\nYou: ").strip()                     # 86
        if user_input.lower() in {"exit", "quit"}:                # 93
            break                                                 # 95
                                                                  #
        conversation.append({"role": "user", "content": user_input})  # 97
        reply = generate_reply(tokenizer, model, conversation)    # 100
        print(reply)                                              # 101
        conversation.append({"role": "assistant", "content": reply})  # 103
```
> **Line 84-103:** The chat loop — conversation list grows with each turn:
> ```
> ┌─── Conversation Memory ────────────────────────────────────────┐
> │                                                                │
> │  Turn 1: [system, user1]                → generate reply1      │
> │  Turn 2: [system, user1, asst1, user2]  → generate reply2      │
> │  Turn 3: [system, …, asst2, user3]      → generate reply3      │
> │                                                                │
> │  The model sees ALL prior turns as context — this is how it   │
> │  "remembers" the conversation.                                 │
> │                                                                │
> │  ⚠️  The list grows forever. Eventually it exceeds the model's │
> │  context window (~8K tokens for SmolLM3) — you'd then need    │
> │  to trim old turns.                                            │
> └────────────────────────────────────────────────────────────────┘
> ```

```python
def main():                                                       # 106
    tokenizer, model = load_model()                               # 107
    chat_loop(tokenizer, model)                                   # 108
                                                                  #
if __name__ == "__main__":                                        # 111
    main()                                                        # 112
```
> **Line 106-112:** Entry point — load model once, then start the chat loop.
> `if __name__ == "__main__"` ensures this only runs when the file is executed directly, not when imported as a module.

---

## Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Causal LM** | Predicts NEXT token (left-to-right), not a fixed label |
| **Autoregressive** | Each generated token feeds back as input for the next |
| **Chat template** | Model-specific format for role-based conversations |
| **temperature** | Randomness knob: 0 = deterministic, 0.7 = balanced, 1.5+ = wild |
| **top_p** | Only sample from the most probable subset of tokens |
| **float16** | Half-precision weights — halves memory, nearly same quality |
| **Context window** | Max tokens the model can "see" at once (~8K for SmolLM3) |
