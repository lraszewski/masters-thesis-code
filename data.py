import torch
import pandas as pd
import numpy as np
import ast
import os
from torch.utils.data import Dataset
from os.path import join

class Task(Dataset):

    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]

        input_ids = torch.tensor(row['input_ids'], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(row['attention_mask'], dtype=torch.long, device=self.device)
        label = torch.tensor(row['label'], dtype=torch.long, device=self.device)

        return input_ids, attention_mask, label


class TripletTask(Dataset):
    
    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device
        self.positive_samples = data[data['label'] == 1]
        self.negative_samples = data[data['label'] == 0]

    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, index):
        anchor = self.positive_samples.iloc[index]

        # randomly select a positive that is not the anchor
        positive = self.positive_samples.sample(n=1).iloc[0]
        while np.array_equal(positive['input_ids'], anchor['input_ids']):
            positive = self.positive_samples.sample(n=1).iloc[0]

        # randomly select a negative
        negative = self.negative_samples.sample(n=1).iloc[0]

        anchor_input_ids = torch.tensor(anchor['input_ids'], dtype=torch.long, device=self.device)
        anchor_attention_mask = torch.tensor(anchor['attention_mask'], dtype=torch.long, device=self.device)

        positive_input_ids = torch.tensor(positive['input_ids'], dtype=torch.long, device=self.device)
        positive_attention_mask = torch.tensor(positive['attention_mask'], dtype=torch.long, device=self.device)

        negative_input_ids = torch.tensor(negative['input_ids'], dtype=torch.long, device=self.device)
        negative_attention_mask = torch.tensor(negative['attention_mask'], dtype=torch.long, device=self.device)

        return (
            anchor_input_ids, anchor_attention_mask,
            positive_input_ids, positive_attention_mask,
            negative_input_ids, negative_attention_mask
        )
    

class ContrastiveTask(Dataset):
    
    def __init__(self, data, device='cuda'):
        self.data = data
        self.device = device
        self.positive_samples = data[data['label'] == 1]

    def __len__(self):
        return len(self.positive_samples)
    
    def __getitem__(self, index):
        anchor = self.positive_samples.iloc[index]
        
        # randomly select another sample
        query = self.positive_samples.sample(n=1).iloc[0]
        while np.array_equal(anchor['input_ids'], query['input_ids']):
            query = self.positive_samples.sample(n=1).iloc[0]

        anchor_input_ids = torch.tensor(anchor['input_ids'], dtype=torch.long, device=self.device)
        anchor_attention_mask = torch.tensor(anchor['attention_mask'], dtype=torch.long, device=self.device)

        query_input_ids = torch.tensor(query['input_ids'], dtype=torch.long, device=self.device)
        query_attention_mask = torch.tensor(query['attention_mask'], dtype=torch.long, device=self.device)

        label = torch.tensor(1 if anchor['label'] == query['label'] else 0, dtype=torch.long, device=self.device)

        return (
            anchor_input_ids, anchor_attention_mask,
            query_input_ids, query_attention_mask,
            label
        )


class TaskDistribution(Dataset):

    def __init__(self, directory, stats,
                 device='cuda',
                 max_tasks=None, 
                 min_puppetmaster=0,
                 min_sockpuppet=0,
                 min_ratio=0,
                 val_split=0.8
                 ):
        
        self.directory = directory
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.min_puppetmaster = min_puppetmaster
        self.min_sockpuppet = min_sockpuppet
        self.min_ratio = min_ratio
        self.val_split = val_split

        stats = stats[stats['num_puppetmaster'] >= min_puppetmaster]
        stats = stats[stats['num_sockpuppets'] >= min_sockpuppet]
        stats = stats[stats['num_negatives'] >= stats['num_positives'] * min_ratio]

        if (max_tasks is not None):
            stats = stats.head(max_tasks)
        
        self.tasks = stats['name'].to_list()
        self.tasks = [join(directory, t) for t in self.tasks]

    def __len__(self):
        return len(self.tasks)
    
    def get_tasks(self):
        return self.tasks
    
    def __getitem__(self, index):

        # read in data
        fn = self.tasks[index]
        data = pd.read_csv(fn, lineterminator='\n')
        data = data.sample(frac=1, random_state=64).reset_index(drop=True)
        fn = os.path.basename(fn)

        # interpret lists
        data['input_ids'] = data['input_ids'].apply(ast.literal_eval)
        data['attention_mask'] = data['attention_mask'].apply(ast.literal_eval)

        # split into relevant classes
        puppetmaster_samples = data[data['label'] == 0]
        sockpuppet_samples = data[data['label'] == 1]
        negatives = data[data['label'] == 2]
        
        # verify task meets requirements
        if (len(puppetmaster_samples) < self.min_puppetmaster):
            raise ValueError(f'{fn}: insufficient puppetmaster samples')
        if (len(sockpuppet_samples) < self.min_sockpuppet):
            raise ValueError(f'{fn}: insufficient sockpuppet samples')
        if (len(negatives) < (len(sockpuppet_samples) + len(puppetmaster_samples)) * self.min_ratio):
            raise ValueError(f'{fn}: insufficient negative samples')

        # process into support (puppetmaster) and test (sockpuppets) sets
        ratio = len(puppetmaster_samples) / (len(puppetmaster_samples) + len(sockpuppet_samples))
        split = int(len(negatives) * ratio)
        support_negatives = negatives.iloc[:split].reset_index(drop=True)
        test_negatives = negatives.iloc[split:].reset_index(drop=True)
        support_data = pd.concat([puppetmaster_samples, support_negatives], axis=0).reset_index(drop=True)
        test_data = pd.concat([sockpuppet_samples, test_negatives], axis=0).reset_index(drop=True)

        # recast labels dealing with old schema
        # 0 is puppetmaster (positive)
        # 1 is sockpuppet (positive)
        # 2 is negative (negative)
        support_data['label'] = support_data['label'].map({0: 1, 1: 1, 2: 0})
        test_data['label'] = test_data['label'].map({0: 1, 1: 1, 2: 0})

        # also create a train and val from the support set
        # ensure the pos neg ratio is preserved (need pos for triplet loss)
        train_neg_split = int(len(support_negatives) * self.val_split)
        train_pos_split = int(len(puppetmaster_samples) * self.val_split)
        train_negs = support_negatives[:train_neg_split].reset_index(drop=True)
        val_negs = support_negatives[train_neg_split:].reset_index(drop=True)
        train_pos = puppetmaster_samples[:train_pos_split].reset_index(drop=True)
        val_pos = puppetmaster_samples[train_pos_split:].reset_index(drop=True)

        train_data = pd.concat([train_pos, train_negs], axis=0).reset_index(drop=True)
        val_data = pd.concat([val_pos, val_negs], axis=0).reset_index(drop=True)
        train_data['label'] = train_data['label'].map({0: 1, 1: 1, 2: 0})
        val_data['label'] = val_data['label'].map({0: 1, 1: 1, 2: 0})

        # compute pos weight based on train data
        n_pos = len(support_data[support_data['label'] == 1])
        n_neg = len(support_data[support_data['label'] == 0])
        pos_weight = torch.tensor(n_neg/n_pos, device=self.device)

        # used as the support set for meta-learning tasks (train + val)
        support_set_standard = Task(support_data, self.device)
        support_set_triplet = TripletTask(support_data, self.device)
        
        # used for training of test distribution tasks
        train_set_standard = Task(train_data, self.device)
        train_set_triplet = TripletTask(train_data, self.device)
        
        # used for validation of test distribution tasks
        val_set_standard = Task(val_data, self.device)
        val_set_triplet = TripletTask(val_data, self.device)

        # used for testing the task
        test_set_standard = Task(test_data, self.device)
        test_set_triplet = TripletTask(test_data, self.device)

        return {
            'fn': fn,
            'pos_weight': pos_weight,

            'support_set_standard': support_set_standard,
            'support_set_triplet': support_set_triplet,

            'train_set_standard': train_set_standard,
            'train_set_triplet': train_set_triplet,

            'val_set_standard': val_set_standard,
            'val_set_triplet': val_set_triplet,

            'test_set_standard': test_set_standard,
            'test_set_triplet': test_set_triplet
        }