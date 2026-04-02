import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Using a Transformer network is simple")
print(encoded_input)

tokens = tokenizer.tokenize("Using a Transformer network is simple")
print(tokens)
input_ids = tokenizer.convert_tokens_to_ids(tokens)
print(input_ids)

decoded_output = tokenizer.decode(encoded_input.input_ids)
print(decoded_output)