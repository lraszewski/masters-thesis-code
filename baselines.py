import torch
from tqdm import tqdm

from helpers import get_distributions, results_folder, save_results, completed

SAVE_DIR = '/mnt/d/results'
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

        random_baseline(fn, test_labels)
        majority_baseline(fn, train_labels, test_labels)
    

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions()
    baselines(test_distribution)