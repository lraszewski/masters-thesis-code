import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import optuna
import time

from datetime import datetime
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoModel
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score
from scipy.interpolate import interp1d
from tqdm import tqdm
from torchmetrics.functional.classification import binary_auroc

from models import EncoderModel, EncoderClassifier, RobertaClassifier
from helpers import EarlyStopper, get_distributions, get_roberta, results_folder, save_results, completed, get_batch_size
from loss import ContrastiveLoss
from training import model_loop, classifier_loop, test
from maml import maml
from reptile import reptile

DEVICE = 'cuda'
TRIPLET = True
SAVE_DIR = './results'

# function to run an experiment. For every task, trains a model, then a
# classifier, and saves predictions 
def experiment(name, task_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, logging=False):
    
    results = results_folder(SAVE_DIR, name)
    roberta = get_roberta()
    
    iterator = task_distribution
    if not logging:
        iterator = tqdm(task_distribution, desc=name)

    for fn, support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in iterator:
        
        # skip tasks for which we already have results
        # assume the parameters of the experiment are unchanged
        if completed(results, fn):
            continue
        
        # create necessary dataloaders
        support_standard_batch_size = get_batch_size(len(support_set_standard))
        query_standard_batch_size = get_batch_size(len(query_set_standard))
        support_standard_dataloader = DataLoader(support_set_standard, batch_size=support_standard_batch_size, shuffle=True, drop_last=True)
        query_standard_dataloader = DataLoader(query_set_standard, batch_size=query_standard_batch_size, shuffle=False, drop_last=False)
        
        mdl_clone = None

        if mdl:

            # model specific dataloaders
            support_triplet_batch_size = get_batch_size(len(support_set_triplet))
            query_triplet_batch_size = get_batch_size(len(query_set_triplet))
            support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=support_triplet_batch_size, shuffle=True, drop_last=True)
            query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=query_triplet_batch_size, shuffle=False, drop_last=False)

            # create a clone of the model and train
            mdl_clone = mdl.clone()
            mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
            model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, support_triplet_dataloader, query_triplet_dataloader, mdl_epochs, logging)

        # create a clone of the classifier and train
        clf_clone = clf.clone()
        clf_optimiser = torch.optim.Adam(clf_clone.parameters(), lr=clf_lr)
        classifier_loop(roberta, mdl_clone, clf_clone, clf_optimiser, clf_criterion, support_standard_dataloader, query_standard_dataloader, clf_epochs, logging)
        
        # get results and save
        labels, probs, embeds = test(roberta, mdl_clone, clf_clone, query_standard_dataloader)
        save_results(results, fn, labels, probs, embeds)

        # track auroc
        auroc = binary_auroc(probs, labels, thresholds=None).item()
        if logging:
            print(auroc)
        else:
            iterator.set_postfix({
                'Task': fn,
                'AUROC': auroc
            })

# tests roberta embeddings + classifier without meta-learning
def roberta_baseline(test_distribution):
    name = "roberta_baseline"
    logging = False
    
    # classifier params
    clf = RobertaClassifier(768, 0.34).to(DEVICE)
    clf_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    clf_epochs = 10
    clf_lr = 0.01
    
    experiment(name, test_distribution, None, None, None, None, clf, clf_criterion, clf_epochs, clf_lr, logging)

# tests roberta embeddings + encoder + classifier without meta-learning
def dual_baseline(test_distribution):
    name = "dual_baseline"
    logging = False

    # model params
    mdl = EncoderModel(768, 2, 6).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    mdl_epochs = 10
    mdl_lr = 0.0001

    # classifier params
    clf = EncoderClassifier(768, 0.34).to(DEVICE)
    clf_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    clf_epochs = 10
    clf_lr = 0.01
    
    experiment(name, test_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, logging)

# tests roberta embeddings + encoder + classifier with reptile meta-learning
def dual_reptile(train_distribution, test_distribution):
    name = "dual_reptile"
    logging = False
    save_path = results_folder(SAVE_DIR, name)

    # model params
    mdl = EncoderModel(768, 2, 6).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    mdl_epochs = 10
    mdl_lr = 0.0001

    # classifier params
    clf = EncoderClassifier(768, 0.34).to(DEVICE)
    clf_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    clf_epochs = 10
    clf_lr = 0.01

    # reptile params
    rep_epochs = 5
    rep_interp = 0.2
    rep_inner_steps = 5

    reptile(save_path, train_distribution, mdl, mdl_criterion, rep_epochs, rep_interp, mdl_lr, rep_inner_steps)
    experiment(name, test_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, logging)


def maml_exp(task_distribution):
    model = EncoderModel(768, 2, 6).to(DEVICE)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    maml(task_distribution, model, criterion, 5, 0.01, 0.05)

def reptile_exp(task_distribution):
    model = EncoderModel(768, 2, 6).to(DEVICE)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    reptile(task_distribution, model, criterion, 10, 0.2, 0.0001, 5)

def reptile_poc(train_distribution, test_distribution):
    model = EncoderModel(768, 2, 6).to(DEVICE)
    classifier = EncoderClassifier(768, 0.2).to(DEVICE)
    metamodel = EncoderModel(768, 2, 6).to(DEVICE)
    criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    clf_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1))
    reptile(train_distribution, metamodel, criterion, 1, 0.2, 0.0001, 5)

    print("---\nmodel:")
    name="dual"
    experiment(name, test_distribution, model, criterion, 1, 0.0001, classifier, clf_criterion, 3, 0.01, False)

    print("---\nmetamodel:")
    name="rep_dual"
    experiment(name, test_distribution, metamodel, criterion, 1, 0.0001, classifier, clf_criterion, 3, 0.01, False)

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions(max_tasks=10)
    # reptile_poc(train_distribution, test_distribution)
    # roberta_baseline(test_distribution)
    dual_reptile(train_distribution, test_distribution)