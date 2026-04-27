import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
import evaluate
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(examples):
    print(f"examples structure: {repr(examples)}")
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)

print(f"raw datasets structure: {raw_datasets}")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"tokenized datasets structure: {tokenized_datasets}")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments('dist/models', eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    print('running compute_metrics')
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()