import os
import torch
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModelForSequenceClassification, AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)

print(f"model_inputs: {model_inputs}")

multiple_sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "So have I!"
]
model_inputs = tokenizer(multiple_sequences)

print(f"model_inputs: {model_inputs}")

# Will pad the sequences up to the maximum sequence length
model_inputs = tokenizer(multiple_sequences, padding="longest")
print(f"model_inputs, longest padding: {model_inputs}")

# Will pad the sequences up to the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(multiple_sequences, padding="max_length")
print(f"model_inputs, max length padding: {model_inputs}")

# Will pad the sequences up to the specified max length
model_inputs = tokenizer(multiple_sequences, padding="max_length", max_length=8)
print(f"model_inputs, specified max length padding: {model_inputs}")

# Will truncate the sequences that are longer than the model max length
# (512 for BERT or DistilBERT)
model_inputs = tokenizer(multiple_sequences, truncation=True)
print(f"model_inputs, truncation: {model_inputs}")

# Will truncate the sequences that are longer than the specified max length
model_inputs = tokenizer(multiple_sequences, max_length=4, truncation=True)
print(f"model_inputs, specified max length truncation: {model_inputs}")

# Returns PyTorch tensors
model_inputs = tokenizer(multiple_sequences, padding=True, return_tensors="pt")
print(f"model_inputs, PyTorch tensors: {model_inputs}")

# Returns NumPy arrays
model_inputs = tokenizer(multiple_sequences, padding=True, return_tensors="np")
print(f"model_inputs, NumPy arrays: {model_inputs}")

sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(tokenizer.decode(model_inputs["input_ids"]))

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(tokenizer.decode(ids))

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(predictions)