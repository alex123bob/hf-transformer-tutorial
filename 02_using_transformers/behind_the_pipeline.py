import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch

classifier = pipeline("sentiment-analysis")
sentiment_result = classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)

print(sentiment_result)

checkpoint = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]

inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
first_token_vector = outputs.last_hidden_state[0, 0, :]
print('first sentence and first token and all hidden vector: ', first_token_vector)
vector_as_list = first_token_vector.tolist()
print('first sentence and first token and all hidden vector as list: ', vector_as_list)

sequence_classifier = AutoModelForSequenceClassification.from_pretrained(checkpoint)
classification_outputs = sequence_classifier(**inputs)
print('classification outputs: ', classification_outputs)
predictions = torch.nn.functional.softmax(classification_outputs.logits, dim=-1)
print('predictions: ', predictions)

print(sequence_classifier.config.id2label)