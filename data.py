import torch
import pandas as pd
import numpy as np
import ast
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
                 puppetmaster=True,
                 max_tasks=None, 
                 min_puppetmaster=0,
                 min_sockpuppet=0,
                 min_ratio=0,
                 split_ratio=0.8
                 ):
        
        self.directory = directory
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.puppetmaster = puppetmaster
        self.split_ratio = split_ratio

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
        fn = self.tasks[index]
        data = pd.read_csv(fn, lineterminator='\n')
        data = data.sample(frac=1, random_state=64).reset_index(drop=True)

        data['input_ids'] = data['input_ids'].apply(ast.literal_eval)
        data['attention_mask'] = data['attention_mask'].apply(ast.literal_eval)
        
        # process into support set and query set
        if (self.puppetmaster):
            puppetmaster_samples = data[data['label'] == 0]
            sockpuppet_samples = data[data['label'] == 1]
            negatives = data[data['label'] == 2]

            if (len(puppetmaster_samples) < 1):
                raise ValueError(f'{fn}: insufficient puppetmaster samples')
            if (len(sockpuppet_samples) < 1):
                raise ValueError(f'{fn}: insufficient sockpuppet samples')
            if (len(negatives) < 2):
                raise ValueError(f'{fn}: insufficient negative samples')

            ratio = len(puppetmaster_samples) / (len(puppetmaster_samples) + len(sockpuppet_samples))
            split = int(len(negatives) * ratio)
            support_negatives = negatives.iloc[:split].reset_index(drop=True)
            query_negatives = negatives.iloc[split:].reset_index(drop=True)
            support_data = pd.concat([puppetmaster_samples, support_negatives], axis=0).reset_index(drop=True)
            query_data = pd.concat([sockpuppet_samples, query_negatives], axis=0).reset_index(drop=True)
        else:
            split = int(len(data) * self.split_ratio)
            support_data = data.iloc[:split].reset_index(drop=True)
            query_data = data.iloc[split:].reset_index(drop=True)

        # recast labels
        support_data['label'] = support_data['label'].map({0: 1, 1: 1, 2: 0})
        query_data['label'] = query_data['label'].map({0: 1, 1: 1, 2: 0})

        support_set_standard = Task(support_data, self.device)
        support_set_triplet = TripletTask(support_data, self.device)
        query_set_standard = Task(query_data, self.device)
        query_set_triplet = TripletTask(query_data, self.device)

        return support_set_standard, support_set_triplet, query_set_standard, query_set_triplet