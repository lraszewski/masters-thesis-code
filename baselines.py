import os
import random
import torch
import pandas as pd
from tqdm import tqdm

from helpers import get_distributions, results_folder, save_results, completed, read_investigation

SAVE_DIR = '/mnt/d/results'
INVS_DIR = './investigations'
NUM_CLASSES = 2
EMBEDDING_SIZE = 768
LABEL_INDEX = 2

# get a set of random predictions
def random_baseline(fn, test_labels):
    results = results_folder(SAVE_DIR, 'random_baseline')
    if completed(results, fn):
        return

    probs = torch.randint(0, NUM_CLASSES, (len(test_labels),))
    embeds = torch.rand(len(test_labels), EMBEDDING_SIZE)
    save_results(results, fn, test_labels, probs, embeds)

# get a set of majority predictions based on the train set
def majority_baseline(fn, train_labels, test_labels):
    results = results_folder(SAVE_DIR, 'majority_baseline')
    if completed(results, fn):
        return
    
    majority_class = torch.mode(train_labels).values.item()
    probs = torch.full_like(test_labels, majority_class)
    embeds = torch.full((len(test_labels), EMBEDDING_SIZE), majority_class)
    save_results(results, fn, test_labels, probs, embeds)

def optimal_baseline(fn):
    results = results_folder(SAVE_DIR, 'optimal_baseline')
    if completed(results, fn):
        return

    path = os.path.join(INVS_DIR, fn)
    data = read_investigation(path)

    # recompute puppetmaster
    puppetmaster = data[data['sock'] == 1]['user'].mode()[0]
    labels = [0 if user == puppetmaster else 1 if sock == 1 else 2 for user, sock in zip(data['user'], data['sock'])]
    data['label'] = labels
    
    # copy the same approach as in data distribution to create test set.
    data = data.sample(frac=1, random_state=64).reset_index(drop=True)

    puppetmaster_samples = data[data['label'] == 0]
    sockpuppet_samples = data[data['label'] == 1]
    negatives = data[data['label'] == 2]

    ratio = len(puppetmaster_samples) / (len(puppetmaster_samples) + len(sockpuppet_samples))
    split = int(len(negatives) * ratio)
    # support_negatives = negatives.iloc[:split].reset_index(drop=True)
    test_negatives = negatives.iloc[split:].reset_index(drop=True)
    # support_data = pd.concat([puppetmaster_samples, support_negatives], axis=0).reset_index(drop=True)
    test_data = pd.concat([sockpuppet_samples, test_negatives], axis=0).reset_index(drop=True)
    # support_data['label'] = support_data['label'].map({0: 1, 1: 1, 2: 0})
    test_data['label'] = test_data['label'].map({0: 1, 1: 1, 2: 0})

    test_labels = test_data['label'].to_list()
    probs = [label if (message != "") else random.uniform(0, 1) for label, message in zip(test_data['label'], test_data['message'])]
    embeds = [torch.full((EMBEDDING_SIZE,), label, dtype=torch.float) if (message != "") else torch.rand(EMBEDDING_SIZE, dtype=torch.float) for label, message in zip(test_data['label'], test_data['message'])]

    test_labels = torch.tensor(test_labels)
    probs = torch.tensor(probs)
    embeds = torch.stack(embeds)

    save_results(results, fn, test_labels, probs, embeds)

# iterate through each task and generate a random and majority baseline
def baselines(test_distribution):

    for task in tqdm(test_distribution):
        
        fn = task['fn']
        train_set = task['train_set_standard']
        test_set = task['test_set_standard']

        train_labels = [train_set[i][LABEL_INDEX] for i in range(len(train_set))]
        test_labels = [test_set[i][LABEL_INDEX] for i in range(len(test_set))]

        train_labels = torch.tensor(train_labels)
        test_labels = torch.tensor(test_labels)

        # random_baseline(fn, test_labels)
        # majority_baseline(fn, train_labels, test_labels)
        optimal_baseline(fn)
    

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions()
    baselines(test_distribution)