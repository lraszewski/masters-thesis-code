import os
import pandas as pd

from transformers import AutoTokenizer
from tqdm import tqdm
from helpers import read_investigation

def tokenize(data, tokenizer, max_length):

    if data.empty:
        raise ValueError("data for encoding is empty")
    
    if data['page'].isnull().all() or data['message'].isnull().all():
        raise ValueError("Data contains null values for 'page' or 'message'")
    
    sentences = [f'{page} {tokenizer.sep_token} {message}' for page, message in zip(data['page'], data['message'])]
    puppetmaster = data[data['sock'] == 1]['user'].mode()[0]
    labels = [0 if user == puppetmaster else 1 if sock == 1 else 2 for user, sock in zip(data['user'], data['sock'])]

    tokenized = tokenizer(
        sentences, 
        return_tensors='pt',
        return_attention_mask=True, 
        padding='max_length', 
        max_length=max_length, 
        truncation=True
    )

    data = pd.DataFrame({
        'input_ids': tokenized['input_ids'].tolist(),
        'attention_mask': tokenized['attention_mask'].tolist(),
        'label': labels
    })

    return data

MODEL_NAME = 'all-distilroberta-v1'
TOKENIZER_NAME = 'sentence-transformers/' + MODEL_NAME
MAX_LENGTH = 128
INPUT_PATH = './investigations'
OUTPUT_PATH = './' + MODEL_NAME + '-tokenized-' + str(MAX_LENGTH)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
fs = [f for f in os.listdir(INPUT_PATH) if f.endswith('csv')]
os.makedirs(OUTPUT_PATH, exist_ok=False)

for f in tqdm(fs):
    data = read_investigation(os.path.join(INPUT_PATH, f))
    data = tokenize(data, tokenizer, MAX_LENGTH)
    data.to_csv(os.path.join(OUTPUT_PATH, f), index=False, lineterminator='\n')