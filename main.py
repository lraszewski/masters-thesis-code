import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from scipy.interpolate import interp1d

from data import TaskDistribution
from models import ClassificationHead, RobertaPooler, EncoderModel, EncoderClassifierModel
from helpers import generate_stats
from loss import ContrastiveLoss

STATS_PATH = 'stats/stats.csv'
INVESTIGATIONS_PATH = 'investigations'
TASKS_PATH = 'all-distilroberta-v1-tokenized-128'
DEVICE = 'cuda'
TRIPLET = True

class Experiment:

    def __init__(self, task_distribution, model, criterion, optimiser, triplet):
        self.task_distribution = task_distribution
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.triplet = triplet
        self.roberta = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1', add_pooling_layer=False).to('cuda').eval()
        for param in self.roberta.parameters():
            param.requires_grad = False

    def run(self):

        for support_set, query_set in self.task_distribution:

            # create a fresh copy of the model to work with
            clone = self.model.clone()
            clone_optimiser = torch.optim.Adam(clone.parameters(), lr=1e-5)

            # train the model on the task
            for epoch in range(10):
                clone.train()
                support_dataloader = DataLoader(support_set, batch_size=10, shuffle=True, drop_last=True)
                train_loss_sum = 0
                pos_train_embeddings = []
                if self.triplet:
                    for (
                        anchor_input_ids, anchor_attention_mask,
                        positive_input_ids, positive_attention_mask,
                        negative_input_ids, negative_attention_mask
                    ) in support_dataloader:
                        anchor_word_embeddings = self.roberta(anchor_input_ids, anchor_attention_mask)[0]
                        positive_word_embeddings = self.roberta(positive_input_ids, positive_attention_mask)[0]
                        negative_word_embeddings = self.roberta(negative_input_ids, negative_attention_mask)[0]
                        anchor_embeddings = clone(anchor_word_embeddings, anchor_attention_mask)
                        positive_embeddings = clone(positive_word_embeddings, positive_attention_mask)
                        negative_embeddings = clone(negative_word_embeddings, negative_attention_mask)
                        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                        loss.backward()
                        clone_optimiser.step()
                        clone_optimiser.zero_grad()
                        train_loss_sum += loss.item()
                        pos_train_embeddings.extend(anchor_embeddings.detach().cpu().numpy())
                else:
                    for input_ids, attention_mask, label in support_dataloader:
                        word_embeddings = self.roberta(input_ids, attention_mask)[0]
                        embeddings = clone(word_embeddings, attention_mask)
                        loss = criterion(embeddings, label)
                        loss.backward()
                        clone_optimiser.step()
                        clone_optimiser.zero_grad()
                        train_loss_sum += loss.item()
                        pos_train_embeddings.extend(embeddings[label == 1].detach().cpu().numpy())

                # train a classifier


                # evaluate the model
                clone.eval()
                all_embeddings = []
                all_labels = []
                query_dataloader = DataLoader(query_set, batch_size=5, shuffle=True, drop_last=False)
                
                for input_ids, attention_mask, label in query_dataloader:
                    word_embeddings = self.roberta(input_ids, attention_mask)[0]
                    embeddings = clone(word_embeddings, attention_mask)
                    all_embeddings.extend(embeddings.detach().cpu().numpy())
                    all_labels.extend(label.detach().cpu().numpy())

                all_embeddings = torch.tensor(all_embeddings)
                all_labels = torch.tensor(all_labels)

                pos_train_embeddings = torch.tensor(pos_train_embeddings)
                positive_centroid = torch.mean(pos_train_embeddings, dim=0)

                positive_centroid = F.normalize(positive_centroid, dim=0)
                all_embeddings = F.normalize(all_embeddings, dim=1)

                similarities = F.cosine_similarity(positive_centroid, all_embeddings).detach().cpu()

                # Precision-Recall Curve
                precision, recall, pr_thresholds = precision_recall_curve(all_labels.numpy(), similarities.numpy())
                pr_auc = auc(recall, precision)
                
                beta = 0.5
                f_beta_scores = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-10)
                best_threshold_idx = np.argmax(f_beta_scores)
                best_threshold = pr_thresholds[best_threshold_idx]
                best_f_beta_score = f_beta_scores[best_threshold_idx]

                predictions = (similarities.numpy() >= best_threshold).astype(int)

                result = {
                    'train_loss': train_loss_sum,
                    # 'accuracy': accuracy_score(all_labels, predictions),
                    # 'precision': precision_score(all_labels, predictions, zero_division=0),
                    # 'recall': recall_score(all_labels, predictions, zero_division=0),
                    # 'f1': fbeta_score(all_labels, predictions, beta=1, zero_division=0),
                    # 'f0.5': fbeta_score(all_labels, predictions, beta=0.5, zero_division=0),
                    'roc_auc': roc_auc_score(all_labels, similarities),
                    # 'pr_auc': pr_auc
                }

                print(result)

            print('---------------------')
        

def experiment(task_distribution, model, model_criterion, classifier, classifier_criterion, epochs, loss_type):
    
    # instantiate a frozen roberta model
    roberta = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1', add_pooling_layer=False).to('cuda').eval()
    for param in roberta.parameters():
        param.requires_grad = False
    
    for support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in task_distribution:
        
        # create a clone of the model
        clone = model.clone()
        clone_optimiser = torch.optim.Adam(clone.parameters(), lr=1e-5)

        # create a classifier
        clf = classifier.clone()
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=1e-3)

        # train model
        for epoch in range(epochs):
            support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=5, shuffle=True, drop_last=True)
            query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=5, shuffle=True, drop_last=False)

            model_train_loss = train_model(roberta, clone, clone_optimiser, support_triplet_dataloader, model_criterion, loss_type)
            model_val_loss = validate_model(roberta, model, query_triplet_dataloader, model_criterion, loss_type)
            
            print(str(epoch) + ": ", model_train_loss, model_val_loss)

        print("")

        # train classifier
        for epoch in range(epochs):
            support_standard_dataloader = DataLoader(support_set_standard, batch_size=5, shuffle=True, drop_last=True)
            query_standard_dataloader = DataLoader(query_set_standard, batch_size=5, shuffle=True, drop_last=False)

            clf_train_loss = train_classifier(roberta, clone, clf, clf_optimiser, support_standard_dataloader, classifier_criterion)            
            clf_val_loss = validate_classifier(roberta, model, clf, query_standard_dataloader, classifier_criterion)

            print(str(epoch) + ": ", clf_train_loss, clf_val_loss)

        metrics = validate(roberta, clone, query_standard_dataloader, clf)
        print("Metrics: " + str(metrics))

        print("---")


def train_model(roberta, model, optimiser, dataloader, criterion, loss_type="triplet"):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()

        if loss_type == "triplet":
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
            
            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask)[0]
            positive_emb = roberta(positive_input_ids, positive_attention_mask)[0]
            negative_emb = roberta(negative_input_ids, negative_attention_mask)[0]

            anchor_emb = model(anchor_emb, anchor_attention_mask)
            positive_emb = model(positive_emb, positive_attention_mask)
            negative_emb = model(negative_emb, negative_attention_mask)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        elif loss_type == "contrastive":
            anchor_input_ids, anchor_attention_mask, query_input_ids, query_attention_mask, labels = batch

            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask)[0]
            query_emb = roberta(query_input_ids, query_attention_mask)[0]

            anchor_emb = model(anchor_emb, anchor_attention_mask)
            query_emb = model(query_emb, query_attention_mask)

            loss = criterion(anchor_emb, query_emb, labels)

        else:
            raise ValueError("invalid loss type")

        loss.backward()
        optimiser.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate_model(roberta, model, dataloader, criterion, loss_type):
    model.eval()
    total_loss = 0.0

    for batch in dataloader:

        if loss_type == "triplet":
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
            
            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask)[0]
            positive_emb = roberta(positive_input_ids, positive_attention_mask)[0]
            negative_emb = roberta(negative_input_ids, negative_attention_mask)[0]

            anchor_emb = model(anchor_emb, anchor_attention_mask)
            positive_emb = model(positive_emb, positive_attention_mask)
            negative_emb = model(negative_emb, negative_attention_mask)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        elif loss_type == "contrastive":
            anchor_input_ids, anchor_attention_mask, query_input_ids, query_attention_mask, labels = batch

            anchor_emb = roberta(anchor_input_ids, anchor_attention_mask)[0]
            query_emb = roberta(query_input_ids, query_attention_mask)[0]

            anchor_emb = model(anchor_emb, anchor_attention_mask)
            query_emb = model(query_emb, query_attention_mask)

            loss = criterion(anchor_emb, query_emb, labels)

        else:
            raise ValueError("invalid loss type")

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_classifier(roberta, model, classifier, optimiser, dataloader, criterion):
    model.eval()
    classifier.train()
    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()
        
        input_ids, attention_mask, labels = batch
        roberta_embeddings = roberta(input_ids, attention_mask)[0]
        model_embeddings = model(roberta_embeddings, attention_mask)
        logits = classifier(model_embeddings)
        loss = criterion(logits.float(), labels.unsqueeze(1).float())

        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
        
def validate_classifier(roberta, model, classifier, dataloader, criterion):
    model.eval()
    classifier.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        roberta_embeddings = roberta(input_ids, attention_mask)[0]
        model_embeddings = model(roberta_embeddings, attention_mask)
        logits = classifier(model_embeddings)
        loss = criterion(logits.float(), labels.unsqueeze(1).float())
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(roberta, model, dataloader, classifier):
    model.eval()
    classifier.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            roberta_embeddings = roberta(input_ids, attention_mask)[0]
            model_embeddings = model(roberta_embeddings, attention_mask)
            probs = classifier(model_embeddings)
            preds = (probs >= 0.5).int()
            all_labels.append(labels)
            all_preds.append(preds)
    
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    result = metrics(all_labels, all_preds)
    return result


def metrics(labels, preds):
    labels = labels.cpu()
    preds = preds.cpu()
    print(preds)
    result = {
        # 'train_loss': train_loss_sum,
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': fbeta_score(labels, preds, beta=1, zero_division=0),
        'f0.5': fbeta_score(labels, preds, beta=0.5, zero_division=0),
        # 'roc_auc': roc_auc_score(labels, similarities),
        # 'pr_auc': pr_auc
    }
    return result


if not os.path.exists(STATS_PATH):
    generate_stats(INVESTIGATIONS_PATH)
stats = pd.read_csv(STATS_PATH)

task_distribution = TaskDistribution(
    directory=TASKS_PATH,
    stats=stats,
    device=DEVICE,
    puppetmaster=True,
    max_tasks=None,
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




#model = RobertaPooler()

# criterion = ContrastiveLoss().to(DEVICE)
# criterion = BatchAllTripletLoss().to(DEVICE)
# exp = Experiment(test_distribution, model, criterion, optimiser, TRIPLET)
# exp.run()

# model = EncoderModel(768, 8, 6).to(DEVICE)
# criterion = nn.TripletMarginLoss()
# experiment(test_distribution, model, criterion, 10, "triplet")

model = EncoderModel(768, 8, 6).to(DEVICE)
model_criterion = nn.TripletMarginLoss()
classifier = ClassificationHead(768, 128).to(DEVICE)
classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0))

experiment(test_distribution, model, model_criterion, classifier, classifier_criterion, 2, "triplet")