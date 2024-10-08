import torch
import torch.nn as nn
import torch.nn.functional as F

class CentroidStrategy:

    def __init__(self, train_embeddings, train_labels):
        mask = train_labels == 1
        positive_centroid = train_embeddings[mask].sum(dim=0) / mask.sum().item()
        positive_centroid = F.normalize(positive_centroid, dim=0)

        self.positive_centroid = positive_centroid
    
    def classify(self, embedding):
        
    