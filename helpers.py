import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModel
from datetime import datetime

from data import TaskDistribution

STATS_PATH = 'stats/stats.csv'
INVESTIGATIONS_PATH = 'investigations'
TASKS_PATH = 'all-distilroberta-v1-tokenized-128'
DEVICE = 'cuda'

# function to read investigation csv into a pandas dataframe
def read_investigation(fn):
    data = pd.read_csv(fn, dtype={'page': str, 'message': str}, lineterminator='\n', parse_dates=['timestamp'], date_format="%Y-%m-%dT%H:%M:%S%z")
    data['message'] = data['message'].fillna('')
    data['page'] = data['page'].fillna('')
    return data

# function to generate task specific stats
def stats(fn, directory):
    data = read_investigation(os.path.join(directory, fn))

    pos_samples = data[(data['sock'] == 1)]
    neg_samples = data[(data['sock'] == 0)]

    puppetmaster = pos_samples['user'].mode()[0]
    puppetmaster_samples = data[data['user'] == puppetmaster]

    return {
        'name': fn,
        'length': data.shape[0],
        'duration': (pos_samples['timestamp'].max() - pos_samples['timestamp'].min()).total_seconds(),

        'num_positives': pos_samples.shape[0],
        'num_negatives': neg_samples.shape[0],
        'num_puppetmaster': puppetmaster_samples.shape[0],
        'num_sockpuppets': (pos_samples.shape[0] - puppetmaster_samples.shape[0]),

        'min_message_length': data['message'].apply(len).min(),
        'ave_message_length': data['message'].apply(len).mean(),
        'max_message_length': data['message'].apply(len).max(),

        'max_page_length': data['page'].apply(len).max(),
        'ave_page_length': data['page'].apply(len).mean(),
        'min_page_length': data['page'].apply(len).min(),
    }

# function to generate a file containing task specific stats for a distribution
def generate_stats(directory):
    data = [stats(fn, directory) for fn in tqdm(os.listdir(directory), desc='generating stats.csv') if fn.endswith('csv')]
    df = pd.DataFrame(data)
    if not os.path.exists('stats'):
        os.makedirs('stats')
    df.to_csv('stats/stats.csv', index=False)

# function to return train and test task distributions
def get_distributions(max_tasks=None):
    if not os.path.exists(STATS_PATH):
        generate_stats(INVESTIGATIONS_PATH)
    stats = pd.read_csv(STATS_PATH)

    task_distribution = TaskDistribution(
        directory=TASKS_PATH,
        stats=stats,
        device=DEVICE,
        puppetmaster=True,
        max_tasks=max_tasks,
        min_puppetmaster=10,
        min_sockpuppet=5,
        min_ratio=1,
        split_ratio=0.8
    )

    gen = torch.Generator().manual_seed(64)
    train_size = int(0.9 * len(task_distribution))
    test_size = len(task_distribution) - train_size
    train_distribution, test_distribution = torch.utils.data.random_split(
        task_distribution, [train_size, test_size],
        generator=gen
    )

    return train_distribution, test_distribution

# function to initialise and freeze a roberta transformer
def get_roberta():
    roberta = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1', add_pooling_layer=False).to('cuda').eval()
    for param in roberta.parameters():
        param.requires_grad = False
    return roberta

# function to compute the variable batch size
def get_batch_size(n):
    if n < 16: return 1
    if n < 32: return 2
    if n < 64: return 4
    if n < 128: return 8
    if n < 256: return 16
    else: return 32

# function to instantiate a results folder for predictions
def results_folder(base_dir, name):
    folder_path = os.path.join(base_dir, name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# function to output a set of model results as a csv
def save_results(dir, fn, labels, probs, embeds):
    df = pd.DataFrame({
        'label': labels.cpu().numpy(),
        'probability': probs.cpu().numpy(),
        'embeds': embeds.cpu().numpy().tolist()
    })
    path = os.path.join(dir, fn)
    df.to_csv(path, index=False)

def completed(dir, fn):
    path = os.path.join(dir, fn)
    return os.path.exists(path)

class EarlyStopper:

    def __init__(self, patience=2, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
