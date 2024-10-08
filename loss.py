import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin
    
    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)
        normalized_embeddings = F.normalize(embeddings)
        
        labels_matrix = (labels.unsqueeze(0) != labels.unsqueeze(1)).long()
        pairwise_cosine_similarities = F.cosine_similarity(normalized_embeddings.unsqueeze(1), normalized_embeddings.unsqueeze(0), dim=2)

        mask = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        loss_same_class = (1 - labels_matrix) * (1 - pairwise_cosine_similarities) ** 2
        loss_same_class.masked_fill_(mask, 0)
        loss_diff_class = labels_matrix * F.relu(pairwise_cosine_similarities - self.margin) ** 2
        loss_diff_class.masked_fill_(mask, 0)
        
        loss_total = loss_same_class + loss_diff_class

        num_pairs = batch_size * (batch_size - 1) # remove diagonal
        loss_average = loss_total.sum() / num_pairs

        return loss_average