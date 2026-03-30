import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline
import torch

# Use MPS (Metal Performance Shaders) for Apple Silicon
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Try with a smaller model first to verify it works
transcriber = pipeline(
    task="automatic-speech-recognition", 
    model="openai/whisper-base",
    device=device
)
result = transcriber(
    "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac",
    generate_kwargs={"language": "en"}
)
print(result)