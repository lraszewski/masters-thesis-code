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

from models import EncoderModel
from helpers import EarlyStopper, get_distributions, get_roberta
from loss import ContrastiveLoss
from training import model_loop, classifier_loop, test
from maml import maml
from reptile import reptile

DEVICE = 'cuda'
TRIPLET = True



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
        
        labels, preds = test(roberta, mdl_clone, query_standard_dataloader, clf_clone)
        result = metrics(labels, preds)
        results.append(result)

        if logging:
            print(result)
            print("---")

    return results



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

def maml_exp(task_distribution):
    model = EncoderModel(768, 2, 6).to(DEVICE)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    maml(task_distribution, model, criterion, 5, 0.01, 0.05)

def reptile_exp(task_distribution):
    model = EncoderModel(768, 2, 6).to(DEVICE)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    reptile(task_distribution, model, criterion, 10, 0.2, 0.0001, 5)

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions()
    maml_exp(train_distribution)
    