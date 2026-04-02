import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Force CPU to avoid MPS backend issues on macOS
device = torch.device("cpu")


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)

sequence = "I've been waiting for a HuggingFace course my whole life."

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"ids: {ids}")

tokenized_input = tokenizer(sequence, return_tensors="pt")
print(f"tokenized_input: {tokenized_input}")

input_ids = torch.tensor([ids]).to(device)
print("Input IDs:", input_ids)

output = model(input_ids)
print("Logits:", output.logits)