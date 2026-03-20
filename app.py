import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import pipeline


text_generator = pipeline(task="text-generation", model="Qwen/Qwen2.5-1.5B")
text_generator("the secret to baking a really good cake is ")