import pandas as pd
import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm

from data import TaskDistribution

STATS_PATH = 'stats/stats.csv'
INVESTIGATIONS_PATH = 'investigations'
TASKS_PATH = 'all-distilroberta-v1-tokenized-128'
DEVICE = 'cuda'

def read_investigation(fn):
    data = pd.read_csv(fn, dtype={'page': str, 'message': str}, lineterminator='\n', parse_dates=['timestamp'], date_format="%Y-%m-%dT%H:%M:%S%z")
    data['message'] = data['message'].fillna('')
    data['page'] = data['page'].fillna('')
    return data

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

def generate_stats(directory):
    data = [stats(fn, directory) for fn in tqdm(os.listdir(directory), desc='generating stats.csv') if fn.endswith('csv')]
    df = pd.DataFrame(data)
    if not os.path.exists('stats'):
        os.makedirs('stats')
    df.to_csv('stats/stats.csv', index=False)

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
        min_puppetmaster=5,
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

def plot_embeddings(embeddings, labels, title="Embedding Clusters", use_tsne=True):

    if use_tsne:
        reducer = TSNE(n_components=2, random_state=42)
    else:
        reducer = PCA(n_components=2)
    
    # Project the embeddings to 2D
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create a scatter plot
    plt.figure(figsize=(8, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()




class EarlyStopper:

    def __init__(self, patience=1, min_delta=0):
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
