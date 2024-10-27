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
from helpers import EarlyStopper, get_distributions, get_roberta, results_folder, save_results, completed, get_batch_size, get_dataloader
from loss import ContrastiveLoss
from training import model_loop, classifier_loop, test
from maml import maml
from reptile import reptile

DEVICE = 'cuda'
TRIPLET = True
SAVE_DIR = '/mnt/d/results'
# SAVE_DIR = './results'

# function to run an experiment. For every task, trains a model, then a
# classifier, and saves predictions 
def experiment(name, task_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, logging=False):
    
    results = results_folder(SAVE_DIR, name)
    roberta = get_roberta()
    
    iterator = task_distribution
    if not logging:
        iterator = tqdm(task_distribution, desc=name)

    for task in iterator:

        # skip tasks for which we already have results
        # assume the parameters of the experiment are unchanged
        fn = task['fn']
        if completed(results, fn):
            continue

        torch.cuda.empty_cache()

        pos_weight = task['pos_weight']
        train_set_standard = task['train_set_standard']
        train_set_triplet = task['train_set_triplet']
        val_set_standard = task['val_set_standard']
        val_set_triplet = task['val_set_triplet']
        test_set_standard = task['test_set_standard']

        # create necessary dataloaders
        train_standard_dataloader = get_dataloader(train_set_standard, shuffle=True, drop_last=True)
        val_standard_dataloader = get_dataloader(val_set_standard)
        test_standard_dataloader = get_dataloader(test_set_standard)
        
        mdl_clone = None

        if mdl:

            # model specific dataloaders
            train_triplet_dataloader = get_dataloader(train_set_triplet, shuffle=True, drop_last=True)
            val_triplet_dataloader = get_dataloader(val_set_triplet)

            # create a clone of the model and train
            mdl_clone = mdl.clone()
            mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
            model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, train_triplet_dataloader, val_triplet_dataloader, mdl_epochs, logging)

        # create a clone of the classifier and train
        clf_clone = clf.clone()
        clf_optimiser = torch.optim.Adam(clf_clone.parameters(), lr=clf_lr)
        classifier_loop(roberta, mdl_clone, clf_clone, clf_optimiser, clf_criterion, train_standard_dataloader, val_standard_dataloader, pos_weight, clf_epochs, logging)
        
        # get results and save
        labels, probs, embeds = test(roberta, mdl_clone, clf_clone, test_standard_dataloader)
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

def roberta_classifier_params():
    clf = RobertaClassifier(768, 0.35).to(DEVICE)
    clf_criterion = F.binary_cross_entropy_with_logits
    clf_epochs = 10
    clf_lr = 0.001
    return clf, clf_criterion, clf_epochs, clf_lr

# a single location to initialise encoder model with chosen params
def encoder_model_params():
    mdl = EncoderModel(768, 2, 6).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=0.2)
    mdl_epochs = 10
    mdl_lr = 0.0001
    return mdl, mdl_criterion, mdl_epochs, mdl_lr

# a single location to initialise encoder classifier with chosen params
def encoder_classifier_params():
    clf = EncoderClassifier(768, 0.35).to(DEVICE)
    clf_criterion = F.binary_cross_entropy_with_logits
    clf_epochs = 10
    clf_lr = 0.001
    return clf, clf_criterion, clf_epochs, clf_lr

# tests roberta embeddings + classifier without meta-learning
def roberta_baseline(test_distribution, logging):
    name = "roberta_baseline"

    clf, clf_criterion, clf_epochs, clf_lr = roberta_classifier_params()
    
    experiment(name, test_distribution, None, None, None, None, clf, clf_criterion, clf_epochs, clf_lr, logging)

# tests roberta embeddings + encoder + classifier without meta-learning
def dual_baseline(test_distribution, logging):
    name = "dual_baseline"

    mdl, mdl_criterion, mdl_epochs, mdl_lr = encoder_model_params()
    clf, clf_criterion, clf_epochs, clf_lr = encoder_classifier_params()
    
    experiment(name, test_distribution, mdl, mdl_criterion, mdl_epochs, mdl_lr, clf, clf_criterion, clf_epochs, clf_lr, logging)

# tests roberta embeddings + encoder + classifier with reptile meta-learning
def dual_reptile(train_distribution, test_distribution, logging):
    name = "dual_reptile"
    save_path = results_folder(SAVE_DIR, name)

    mdl, mdl_criterion, mdl_epochs, mdl_lr = encoder_model_params()
    clf, clf_criterion, clf_epochs, clf_lr = encoder_classifier_params()

    # reptile params
    rep_epochs = 10
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
    train_distribution, test_distribution = get_distributions()
    logging = False
    dual_reptile(train_distribution, test_distribution, logging)