import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import tqdm

from main import get_roberta, model_loop, classifier_loop, validate, metrics
from models import EncoderModel, ClassificationHead
from helpers import get_distributions
from torch.utils.data import DataLoader

DEVICE = 'cuda'

def log_tune_result(study, trial):
    trial_data = {
        "trial_number": trial.number,
        "value": trial.value,
        "params": trial.params,
    }
    results_df = pd.DataFrame([trial_data])
    fn = "results/" + study.study_name + ".csv"
    results_df.to_csv(fn, mode='a', header=not pd.io.common.file_exists(fn), index=False)

def model_objective(trial):

    # tuning parameters
    mdl_lr = trial.suggest_float('model_lr', 1e-5, 1e-2)
    nhead = trial.suggest_int('nhead', 2, 8, step=2)
    num_layers = trial.suggest_int('num_layers', 2, 8)
    margin = trial.suggest_float('margin', 0.0, 1.0)

    # constant parameters
    roberta = get_roberta()
    mdl = EncoderModel(768, nhead, num_layers).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=margin
    )
    mdl_epochs = 2
    loss_type = "triplet"
    logging = False

    losses = []
    for support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in tqdm(train_distribution):
        
        support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=10, shuffle=True, drop_last=True)
        query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=10, shuffle=True, drop_last=False)
        
        # create a clone of the model
        mdl_clone = mdl.clone()
        mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
        loss = model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, support_triplet_dataloader, query_triplet_dataloader, mdl_epochs, loss_type, logging)
        losses.append(loss)
    
    return sum(losses) / len(losses)

def tune_model():
    study = optuna.create_study(study_name="model_hyperparameters")
    study.optimize(model_objective, n_trials=50, callbacks=[log_tune_result])
    print("Best model hyperparameters: ", study.best_params)

def train_models_for_inference(train_distribution):

    # hyper params
    nhead = 2
    num_layers = 6
    margin = 0.001
    mdl_lr = 0.005
    mdl_epochs = 10

    # setup
    roberta = get_roberta()
    mdl = EncoderModel(768, nhead, num_layers).to(DEVICE)
    mdl_criterion = nn.TripletMarginWithDistanceLoss(
        distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
        margin=margin
    )
    loss_type = "triplet"
    logging = False

    # train
    mdls = []
    for support_set_standard, support_set_triplet, query_set_standard, query_set_triplet in tqdm(train_distribution, desc="training embedding models"):
        
        support_triplet_dataloader = DataLoader(support_set_triplet, batch_size=10, shuffle=True, drop_last=True)
        query_triplet_dataloader = DataLoader(query_set_triplet, batch_size=10, shuffle=True, drop_last=False)
        mdl_clone = mdl.clone()
        mdl_optimiser = torch.optim.Adam(mdl_clone.parameters(), lr=mdl_lr)
        loss = model_loop(roberta, mdl_clone, mdl_optimiser, mdl_criterion, support_triplet_dataloader, query_triplet_dataloader, mdl_epochs, loss_type, logging)
        mdl_clone.eval()
        mdls.append(mdl_clone)
    
    return mdls



def classifier_objective(trial, models):

    # tuning parameters
    clf_lr = trial.suggest_float('clf_lr', 1e-5, 1e-1)
    dropout = trial.suggest_float('dropout', 0.0, 1.0)
    reduction = trial.suggest_categorical("reduction", ["mean", "sum"])
    n_layers = trial.suggest_int('n_layers', 1, 6)
    pos_weight = trial.suggest_float('pos_weight', 0.5, 5.0)

    # constants
    roberta = get_roberta()
    clf_criterion = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.tensor(pos_weight))
    
    # design classifier
    layers = []
    in_features = 768
    for i in range(n_layers):
        out_features = trial.suggest_int(f'n_units_l{i}', 4, 768)
        layers.append(torch.nn.Linear(in_features, out_features))
        layers.append(torch.nn.GELU())
        layers.append(torch.nn.Dropout(dropout))
        in_features = out_features
    layers.append(torch.nn.Linear(in_features, 1))
    
    losses = []
    results = []
    for i, (support_set_standard, support_set_triplet, query_set_standard, query_set_triplet) in tqdm(enumerate(train_distribution)):
        
        # grab the model
        mdl = models[i]

        support_standard_dataloader = DataLoader(support_set_standard, batch_size=10, shuffle=True, drop_last=True)
        query_standard_dataloader = DataLoader(query_set_standard, batch_size=10, shuffle=True, drop_last=False)
        
        # create a clone of the classifier
        clf = torch.nn.Sequential(*copy.deepcopy(layers)).to(torch.device(DEVICE))
        clf_optimiser = torch.optim.Adam(clf.parameters(), lr=clf_lr)

        loss = classifier_loop(roberta, mdl, clf, clf_optimiser, clf_criterion, support_standard_dataloader, query_standard_dataloader, 2, False)
        losses.append(loss)

        labels, preds = validate(roberta, mdl, query_standard_dataloader, clf)
        result = metrics(labels, preds)
        results.append(result)

    # return sum(losses) / len(losses)
    return sum([r['f0.5'] for r in results]) / len(results)
    

def tune_classifier():
    study = optuna.create_study(study_name="classifier_hyperparameters") #direction="maximize"
    models = train_models_for_inference(train_distribution)
    study.optimize(lambda trial: classifier_objective(trial, models), n_trials=50, callbacks=[log_tune_result])
    print("Best classifier hyperparameters: ", study.best_params)

if __name__ == '__main__':
    train_distribution, test_distribution = get_distributions(max_tasks=10)

    # tune_model()
    tune_classifier()