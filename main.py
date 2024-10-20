import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import optuna
import time

from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from scipy.interpolate import interp1d
from tqdm import tqdm

from models import ClassificationHead, RobertaPooler, EncoderModel, BiLSTMModel, EncoderClassifierModel
from helpers import EarlyStopper, get_distributions, get_roberta
from loss import ContrastiveLoss

DEVICE = 'cuda'
TRIPLET = True

class Experiment:

    def __init__(self, task_distribution, model, criterion, optimiser, triplet):
        self.task_distribution = task_distribution
        self.model = model
        self.criterion = criterion
        self.optimiser = optimiser
        self.triplet = triplet
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
                        anchor_word_embeddings = self.roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
                        positive_word_embeddings = self.roberta(positive_input_ids, positive_attention_mask).last_hidden_state
                        negative_word_embeddings = self.roberta(negative_input_ids, negative_attention_mask).last_hidden_state
                        anchor_embeddings = clone(anchor_word_embeddings, anchor_attention_mask)
                        positive_embeddings = clone(positive_word_embeddings, positive_attention_mask)
                        negative_embeddings = clone(negative_word_embeddings, negative_attention_mask)
                        loss = self.criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
                        loss.backward()
                        clone_optimiser.step()
                        clone_optimiser.zero_grad()
                        train_loss_sum += loss.item()
                        pos_train_embeddings.extend(anchor_embeddings.detach().cpu().numpy())
                else:
                    for input_ids, attention_mask, label in support_dataloader:
                        word_embeddings = self.roberta(input_ids, attention_mask).last_hidden_state
                        embeddings = clone(word_embeddings, attention_mask)
                        loss = self.criterion(embeddings, label)
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
                    word_embeddings = self.roberta(input_ids, attention_mask).last_hidden_state
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
        



def experiment(task_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, loss_type, logging=False):
    
    # instantiate a frozen roberta model
    roberta = get_roberta()
    
    # keep track of results for each task
    results = []
    iterator = task_distribution
    if not logging:
        iterator = tqdm(task_distribution)

    for support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in iterator:
        
        support_standard_dataloader = DataLoader(support_set_standard, batch_size=10, shuffle=True, drop_last=True)
        query_standard_dataloader = DataLoader(query_set_standard, batch_size=10, shuffle=True, drop_last=False)

        support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=10, shuffle=True, drop_last=True)
        query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=10, shuffle=True, drop_last=False)
        
        # create a clone of the model
        mdl_clone = mdl.clone()
        mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
        model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, support_triplet_dataloader, query_triplet_dataloader, mdl_epochs, loss_type, logging)

        if logging:
            print("")

        # create a classifier
        clf_clone = clf.clone()
        clf_optimiser = torch.optim.Adam(clf_clone.parameters(), lr=clf_lr)
        classifier_loop(roberta, mdl_clone, clf_clone, clf_optimiser, clf_criterion, support_standard_dataloader, query_standard_dataloader, clf_epochs, logging)
        
        labels, preds = validate(roberta, mdl_clone, query_standard_dataloader, clf_clone)
        result = metrics(labels, preds)
        results.append(result)

        if logging:
            print(result)
            print("---")

    return results


def model_loop(roberta, model, optimiser, criterion, support_dataloader, query_dataloader, epochs, loss_type, logging):
    train_losses = []
    val_losses = []
    early_stopper = EarlyStopper(patience=2, min_delta=0.01)
    for epoch in range(epochs):

        train_loss = train_model(roberta, model, optimiser, support_dataloader, criterion, loss_type)
        val_loss = validate_model(roberta, model, query_dataloader, criterion, loss_type)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if logging:
            print(str(epoch) + ": ", train_loss, val_loss)

        if early_stopper.early_stop(val_loss):
            break
    
    return min(val_losses)

def train_model(roberta, model, optimiser, dataloader, criterion, loss_type="triplet"):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()

        if loss_type == "triplet":
            anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
            
            # with torch.autocast(device_type=DEVICE):
            with torch.no_grad():
                anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
                positive_emb = roberta(positive_input_ids, positive_attention_mask).last_hidden_state
                negative_emb = roberta(negative_input_ids, negative_attention_mask).last_hidden_state

            anchor_emb = model(anchor_emb, anchor_attention_mask)
            positive_emb = model(positive_emb, positive_attention_mask)
            negative_emb = model(negative_emb, negative_attention_mask)

            loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        elif loss_type == "contrastive":
            anchor_input_ids, anchor_attention_mask, query_input_ids, query_attention_mask, labels = batch

            with torch.no_grad():
                anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
                query_emb = roberta(query_input_ids, query_attention_mask).last_hidden_state

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
        with torch.no_grad():
            if loss_type == "triplet":
                anchor_input_ids, anchor_attention_mask, positive_input_ids, positive_attention_mask, negative_input_ids, negative_attention_mask = batch
                
                anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
                positive_emb = roberta(positive_input_ids, positive_attention_mask).last_hidden_state
                negative_emb = roberta(negative_input_ids, negative_attention_mask).last_hidden_state

                anchor_emb = model(anchor_emb, anchor_attention_mask)
                positive_emb = model(positive_emb, positive_attention_mask)
                negative_emb = model(negative_emb, negative_attention_mask)

                loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            elif loss_type == "contrastive":
                anchor_input_ids, anchor_attention_mask, query_input_ids, query_attention_mask, labels = batch

                anchor_emb = roberta(anchor_input_ids, anchor_attention_mask).last_hidden_state
                query_emb = roberta(query_input_ids, query_attention_mask).last_hidden_state

                anchor_emb = model(anchor_emb, anchor_attention_mask)
                query_emb = model(query_emb, query_attention_mask)

                loss = criterion(anchor_emb, query_emb, labels)

            else:
                raise ValueError("invalid loss type")

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def classifier_loop(roberta, model, classifier, optimiser, criterion, support_dataloader, query_dataloader, epochs, logging):

    train_losses = []
    val_losses = []
    early_stopper = EarlyStopper(patience=2, min_delta=0.1)
    for epoch in range(epochs):

        train_loss = train_classifier(roberta, model, classifier, optimiser, support_dataloader, criterion)            
        val_loss = validate_classifier(roberta, model, classifier, query_dataloader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logging:
            print(str(epoch) + ": ", train_loss, val_loss)

        if early_stopper.early_stop(val_loss):
            break
    
    return min(val_losses)

def train_classifier(roberta, model, classifier, optimiser, dataloader, criterion):
    if model:
        model.eval()
    classifier.train()
    total_loss = 0.0

    for batch in dataloader:
        optimiser.zero_grad()
        
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
        logits = classifier(model_embeddings)
        loss = criterion(logits.float(), labels.unsqueeze(1).float())

        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss
        
def validate_classifier(roberta, model, classifier, dataloader, criterion):
    if model:
        model.eval()
    classifier.eval()
    total_loss = 0.0

    for batch in dataloader:
        input_ids, attention_mask, labels = batch
        with torch.no_grad():
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
            logits = classifier(model_embeddings)
            loss = criterion(logits.float(), labels.unsqueeze(1).float())
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss



def validate(roberta, model, dataloader, classifier):
    if model:
        model.eval()
    classifier.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            roberta_embeddings = roberta(input_ids, attention_mask).last_hidden_state
            if model is None:
                # use the roberta cls token if there is no model
                model_embeddings = roberta_embeddings[:,0,:]
            else:
                model_embeddings = model(roberta_embeddings, attention_mask)
            logits = classifier(model_embeddings)
            probs = torch.sigmoid(logits).flatten()
            all_labels.append(labels)
            all_probs.append(probs)
    
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)
    return all_labels, all_probs

def metrics(labels, preds): #, best_model_val_loss, best_clf_val_loss):
    labels = labels.cpu()
    preds = preds.cpu()
    result = {
        # 'train_loss': train_loss_sum,
        #'best_model_val_loss': best_model_val_loss,
        #'best_clf_val_loss': best_clf_val_loss,
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': fbeta_score(labels, preds, beta=1, zero_division=0),
        'f0.5': fbeta_score(labels, preds, beta=0.5, zero_division=0),
        # 'roc_auc': roc_auc_score(labels, similarities),
        # 'pr_auc': pr_auc
    }
    return result




def single_exp():
    model = EncoderModel(768, 2, 3).to(DEVICE)
    # model = BiLSTMModel(768, int(768/2), 6).to(DEVICE)
    # model = RobertaPooler()
    # model_criterion = nn.TripletMarginLoss(margin=0.002)
    model_criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=0.2864
    )
    model_epochs = 10
    model_lr = 0.005
    classifier = ClassificationHead(768, 0.13).to(DEVICE)
    classifier_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(4.63))
    classifier_epochs = 10
    classifier_lr = 0.003
    results = experiment(train_distribution, model, model_criterion, model_epochs, model_lr, classifier, classifier_criterion, classifier_epochs, classifier_lr, "triplet", logging=True)


if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions()
    single_exp()