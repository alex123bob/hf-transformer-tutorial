import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertModel
from huggingface_hub import notebook_login
import torch

# model = AutoModel.from_pretrained('bert-base-cased')
model = BertModel.from_pretrained("bert-base-cased")

# comment out the following line if you don't want to save the model locally
# model.save_pretrained('./bert-base-cased')

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
decoded_str = tokenizer.decode(encoded_input['input_ids'])
print(decoded_str)

encoded_input = tokenizer(["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt")
print(encoded_input)

encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=7,
    return_tensors="pt",
)
print(encoded_input)

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
encoded_sequences = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
model_inputs = torch.tensor(encoded_sequences.input_ids)
print(model_inputs)

output = model(model_inputs)
print(output)