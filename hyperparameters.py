import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_auroc
import copy

from training import model_loop, classifier_loop, test
from models import EncoderModel
from helpers import get_distributions, get_roberta, get_batch_size
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
    clf_pos_weight = trial.suggest_float('clf_pos_weight', 0.5, 5.0)
    return clf_lr, clf_dropout, clf_n_layers, clf_pos_weight

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
    clf_lr, clf_dropout, clf_n_layers, clf_pos_weight = suggest_classifier_parameters(trial)
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
    for _, pos_weight, support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in train_distribution:

        # create necessary dataloaders
        support_standard_batch_size = get_batch_size(len(support_set_standard))
        support_triplet_batch_size = get_batch_size(len(support_set_triplet))
        query_standard_batch_size = get_batch_size(len(query_set_standard))
        query_triplet_batch_size = get_batch_size(len(query_set_triplet))
        support_standard_dataloader = DataLoader(support_set_standard, batch_size=support_standard_batch_size, shuffle=True, drop_last=True)
        support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=support_triplet_batch_size, shuffle=True, drop_last=True)
        query_standard_dataloader = DataLoader(query_set_standard, batch_size=query_standard_batch_size, shuffle=False, drop_last=False)
        query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=query_triplet_batch_size, shuffle=False, drop_last=False)

        # create model, classifier and optimisers
        mdl_clone = mdl.clone()
        mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
        clf = torch.nn.Sequential(*copy.deepcopy(clf_layers)).to(torch.device(DEVICE))
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=clf_lr)

        # train
        mdl_loss = model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, support_triplet_dataloader, query_triplet_dataloader, mdl_epochs, logging=False)
        clf_loss = classifier_loop(roberta, mdl_clone, clf, clf_optimiser, clf_criterion, support_standard_dataloader, query_standard_dataloader, pos_weight, clf_epochs, logging=False)

        # test
        labels, probs = test(roberta, mdl_clone, clf, query_standard_dataloader)
        auroc = binary_auroc(probs, labels, thresholds=None)
        aurocs.append(auroc.item())
        print(auroc.item())
    
    return sum(aurocs) / len(aurocs)

# objective function to tune the classifier attached to a frozen roberta model
def roberta_classifier_objective(trial, train_distribution):

    # tuned parameters
    clf_lr, clf_dropout, clf_n_layers, clf_pos_weight = suggest_classifier_parameters(trial)
    clf_layers = design_classifier(trial, clf_n_layers, clf_dropout)
    clf_criterion = F.binary_cross_entropy_with_logits()

    # constants
    roberta = get_roberta()
    clf_epochs = EPOCHS

    losses = []
    aurocs = []
    for _, pos_weight, support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in train_distribution:

        # create necessary dataloaders
        support_batch_size = get_batch_size(len(support_set_standard))
        query_batch_size = get_batch_size(len(query_set_standard))
        support_standard_dataloader = DataLoader(support_set_standard, batch_size=support_batch_size, shuffle=True, drop_last=True)
        query_standard_dataloader = DataLoader(query_set_standard, batch_size=query_batch_size, shuffle=False, drop_last=False)

        # create a classifier and optimiser
        clf = torch.nn.Sequential(*copy.deepcopy(clf_layers)).to(torch.device(DEVICE))
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=clf_lr)

        # train
        loss = classifier_loop(roberta, None, clf, clf_optimiser, clf_criterion, support_standard_dataloader, query_standard_dataloader, pos_weight, clf_epochs, False)
        losses.append(loss)

        # test
        labels, probs = test(roberta, None, clf, query_standard_dataloader)
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
    review_study('roberta_classifier_hyperparameters_adam', 'sqlite:///hyper-parameters/roberta_classifier_adam_study.db')