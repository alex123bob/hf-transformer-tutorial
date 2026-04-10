import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from datasets import load_dataset

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

# This is new
batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

raw_datasets = load_dataset("glue", "mrpc")
raw_train_datasets = raw_datasets["train"]
raw_validation_datasets = raw_datasets["validation"]

print(raw_train_datasets.features)

print(f"15 element of training set: {raw_train_datasets[14]}")
print(f"87 element of validation set: {raw_validation_datasets[86]}")

print(f"raw train datasets: {raw_train_datasets['sentence1']} first 10 elements of raw train datasets: {raw_train_datasets['sentence1'][:10]}")
tokenized_sentence_1_of_training_set = tokenizer(raw_train_datasets["sentence1"][:10])
print(f"Tokenized sentence 1 of training set: {tokenized_sentence_1_of_training_set}")
tokenized_sentence_2_of_training_set = tokenizer(raw_train_datasets["sentence2"][:10])
print(f"Tokenized sentence 2 of training set: {tokenized_sentence_2_of_training_set}")

inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(f"paired inputs after tokenization: {inputs}")

# Try it out! Take element 15 of the training set and tokenize the two sentences separately and as a pair. What’s the difference between the two results?
sentence_1 = raw_train_datasets[14]["sentence1"]
sentence_2 = raw_train_datasets[14]["sentence2"]
print(f"sentence 1: {sentence_1}")
print(f"sentence 2: {sentence_2}")
tokenized_sentence_1 = tokenizer(sentence_1, sentence_2)
print(f"Tokenized sentence 1 and 2 as a pair: {tokenized_sentence_1}")
print(f"Decode tokenized sentence 1 and 2 as a pair: {tokenizer.decode(tokenized_sentence_1['input_ids'])}")

tokenized_dataset = tokenizer(
    raw_datasets["train"]["sentence1"][:],
    raw_datasets["train"]["sentence2"][:],
    padding=True,
    truncation=True,
)

print(f"Tokenized dataset: {tokenized_dataset}")