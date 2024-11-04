import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc
import copy

from training import model_loop, classifier_loop, test
from models import EncoderModel
from helpers import get_distributions, get_roberta, get_dataloader
from torch.utils.data import DataLoader

DEVICE = 'cuda'
EPOCHS = 10

# given a trial, return suggested parameters for the model
def suggest_model_parameters(trial):
    mdl_lr = trial.suggest_float('mdl_lr', 1e-5, 1e-2)
    mdl_n_heads = trial.suggest_int('mdl_n_heads', 2, 8, step=2)
    mdl_n_layers = trial.suggest_int('mdl_n_layers', 2, 8)
    mdl_margin = trial.suggest_float('mdl_margin', 0.0, 1.0)
    return mdl_lr, mdl_n_heads, mdl_n_layers, mdl_margin

# given a trial, return suggested parameters for the classifier
def suggest_classifier_parameters(trial):
    clf_lr = trial.suggest_float('clf_lr', 1e-5, 1e-1)
    clf_dropout = trial.suggest_float('clf_dropout', 0.0, 1.0)
    clf_n_layers = trial.suggest_int('clf_n_layers', 1, 6)
    return clf_lr, clf_dropout, clf_n_layers

# given a trial, n_layers and dropout, return a set of classifier layers
def design_classifier(trial, clf_n_layers, clf_dropout):
    layers = []
    in_features = 768
    for i in range(clf_n_layers):
        out_features = trial.suggest_int(f'clf_n_units_l{i}', 4, 768)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(clf_dropout))
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 1))
    return layers

# given a study and completed trial, log the result of the trial
def log_tune_result(study, trial):
    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
    }
    results_df = pd.DataFrame([trial_data])
    fn = "results/" + study.study_name + ".csv"
    results_df.to_csv(fn, mode='a', header=not pd.io.common.file_exists(fn), index=False)

def review_study(name, storage):
    study = optuna.load_study(name, storage=storage)
    print("Best trial value (objective): ", study.best_trial.value)
    print("Best hyperparameters: ", study.best_trial.params)
    optuna.visualization.plot_optimization_history(study).show()
    optuna.visualization.plot_param_importances(study).show()
    optuna.visualization.plot_slice(study).show()
    optuna.visualization.plot_parallel_coordinate(study).show()

# objective function to tune the embedding and classifier models
def dual_objective(trial, train_distribution):

    # classifier tuned parameters
    clf_lr, clf_dropout, clf_n_layers = suggest_classifier_parameters(trial)
    clf_layers = design_classifier(trial, clf_n_layers, clf_dropout)
    clf_criterion = F.binary_cross_entropy_with_logits()

    # model tuned parameters
    mdl_lr, mdl_n_heads, mdl_n_layers, mdl_margin = suggest_model_parameters(trial)
    mdl = EncoderModel(768, mdl_n_heads, mdl_n_layers).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),margin=mdl_margin)

    # constants
    roberta = get_roberta()
    clf_epochs = EPOCHS
    mdl_epochs = EPOCHS

    aurocs = []
    for task in train_distribution:

        # unpack task
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
        train_triplet_dataloader = get_dataloader(train_set_triplet, shuffle=True, drop_last=True)
        val_triplet_dataloader = get_dataloader(val_set_triplet)

        # create model, classifier and optimisers
        mdl_clone = mdl.clone()
        mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
        clf = torch.nn.Sequential(*copy.deepcopy(clf_layers)).to(torch.device(DEVICE))
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=clf_lr)

        # train
        model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, train_triplet_dataloader, val_triplet_dataloader, mdl_epochs, logging=False)
        classifier_loop(roberta, mdl_clone, clf, clf_optimiser, clf_criterion, train_standard_dataloader, val_standard_dataloader, pos_weight, clf_epochs, logging=False)

        # test
        labels, probs = test(roberta, mdl_clone, clf, test_standard_dataloader)
        auroc = binary_auroc(probs, labels, thresholds=None)
        aurocs.append(auroc.item())
        print(auroc.item())
    
    return sum(aurocs) / len(aurocs)

# objective function to tune the classifier attached to a frozen roberta model
def roberta_classifier_objective(trial, train_distribution):

    # tuned parameters
    clf_lr, clf_dropout, clf_n_layers = suggest_classifier_parameters(trial)
    clf_layers = design_classifier(trial, clf_n_layers, clf_dropout)
    clf_criterion = F.binary_cross_entropy_with_logits()

    # constants
    roberta = get_roberta()
    clf_epochs = EPOCHS

    losses = []
    aurocs = []
    for task in train_distribution:

        # unpack task
        pos_weight = task['pos_weight']
        train_set_standard = task['train_set_standard']
        val_set_standard = task['val_set_standard']
        test_set_standard = task['test_set_standard']

        # create necessary dataloaders
        train_standard_dataloader = get_dataloader(train_set_standard, shuffle=True, drop_last=True)
        val_standard_dataloader = get_dataloader(val_set_standard)
        test_standard_dataloader = get_dataloader(test_set_standard)

        # create a classifier and optimiser
        clf = torch.nn.Sequential(*copy.deepcopy(clf_layers)).to(torch.device(DEVICE))
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=clf_lr)

        # train
        loss = classifier_loop(roberta, None, clf, clf_optimiser, clf_criterion, train_standard_dataloader, val_standard_dataloader, pos_weight, clf_epochs, False)
        losses.append(loss)

        # test
        labels, probs = test(roberta, None, clf, test_standard_dataloader)
        auroc = binary_auroc(probs, labels, thresholds=None)
        aurocs.append(auroc.item())
        print(auroc.item())

    return sum(aurocs) / len(aurocs)

# tunes combination of model and classifier
def tune_dual(train_distribution):
    study = optuna.create_study(study_name="dual_hyperparameters", direction="maximize", storage='sqlite:///hyper-parameters/dual_hyperparameters_study.db', load_if_exists=True)
    study.optimize(lambda trial: dual_objective(trial, train_distribution), n_trials=100, callbacks=[log_tune_result])
    print("Best hyperparameters: ", study.best_params)

# tunes a classifier for the frozen roberta model
def tune_roberta_classifier(train_distribution):
    study = optuna.create_study(study_name="roberta_classifier_hyperparameters_adam", direction="maximize", storage='sqlite:///hyper-parameters/roberta_classifier_adam_study.db', load_if_exists=True)
    study.optimize(lambda trial: roberta_classifier_objective(trial, train_distribution), n_trials=100, callbacks=[log_tune_result])
    print("Best hyperparameters: ", study.best_params)

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions(max_tasks=10)
    # tune_roberta_classifier(train_distribution)
    # tune_dual(train_distribution)
    review_study('dual_hyperparameters', 'sqlite:///hyper-parameters/dual_hyperparameters_adam_study.db')