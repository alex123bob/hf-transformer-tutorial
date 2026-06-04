# app.py — Annotated Walkthrough

> **What this script does:** Uses a HuggingFace pipeline to transcribe speech (audio → text) with OpenAI's Whisper model.

```
┌─────────────────────────────────────────────────────────────────┐
│                     HIGH-LEVEL FLOW                              │
│                                                                 │
│   Audio URL ──→ [ pipeline("automatic-speech-recognition") ] ──→ Text │
│                                                                 │
│   One line of code. The pipeline handles everything inside.     │
└─────────────────────────────────────────────────────────────────┘
```

---

```python
import os                                                         # 1
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'              # 2
```
> **Line 1-2:** Set HuggingFace to use a mirror URL. If you're in a region where `huggingface.co` is slow or blocked, this redirects all model downloads through a faster mirror.

```python
from transformers import pipeline                                 # 3
```
> **Line 3:** `pipeline` is the highest-level API in HuggingFace. It bundles a **model + tokenizer + pre/post-processing** into a single callable. You tell it a task name, it does the rest.

```python
import torch                                                      # 4
```
> **Line 4:** PyTorch — the deep learning framework running all neural network computations under the hood.

```python
# Use MPS (Metal Performance Shaders) for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"   # 7
print(f"Using device: {device}")                                  # 8
```
> **Line 7-8:** Pick the fastest available hardware:
> ```
> ┌─────────────────────────────────────────────────┐
> │   Apple Silicon Mac? → "mps" (GPU, ~10x faster) │
> │   Otherwise?         → "cpu" (fallback, slower)  │
> │   NVIDIA machine?    → you'd use "cuda" instead  │
> └─────────────────────────────────────────────────┘
> ```

```python
transcriber = pipeline(                                            # 11
    task="automatic-speech-recognition",                           # 12
    model="openai/whisper-base",                                   # 13
    device=device                                                  # 14
)                                                                  # 15
```
> **Line 11-15:** Create the ASR pipeline:
> | Parameter | Meaning |
> |-----------|---------|
> | `task` | What to do — here: convert speech to text |
> | `model` | Which neural network — Whisper (74M params, trained on 680K hours of audio) |
> | `device` | Where to compute — GPU or CPU |
>
> This downloads ~290MB of model weights the first time.

```python
result = transcriber(                                              # 16
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",  # 17
    generate_kwargs={"language": "en"}                             # 18
)                                                                  # 19
```
> **Line 16-19:** Run transcription on a remote audio file (MLK speech clip).
>
> ```
> ┌──────────────────── What Happens Inside ────────────────────────┐
> │                                                                  │
> │  1. Download .flac audio from URL                                │
> │  2. Decode to waveform: ∿∿∿∿∿∿∿∿ (amplitude over time)          │
> │  3. Convert to Mel Spectrogram (frequency × time grid)           │
> │  4. Feed spectrogram into Whisper neural network                 │
> │  5. Network generates tokens: [I] → [have] → [a] → [dream]...  │
> │  6. Tokens decoded to string                                     │
> │                                                                  │
> └────────────────────────────────────────────────────────────────┘
> ```
> `language="en"` hints that audio is English for better accuracy.

```python
print(result)                                                     # 20
```
> **Line 20:** Output: `{'text': 'I have a dream that one day...'}`

---

## Key Concepts

| Concept | What it is |
|---------|------------|
| **Pipeline** | High-level wrapper: Input → Magic → Output |
| **Model** | Neural net with millions of learned numbers (parameters) |
| **Device** | GPU (fast, parallel math) or CPU (slow, sequential) |
| **Inference** | Using a trained model to make predictions (not training) |
