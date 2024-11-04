import os
import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, accuracy_score, precision_score, recall_score, fbeta_score
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.decomposition import PCA
import seaborn as sns

SAVE_PATH = '/mnt/d/results/'

# compute the metrics of a single investigation
def get_result(file_path):
    data = pd.read_csv(file_path, usecols=['label', 'probability'])

    # auprc
    precision, recall, prc_thresholds = precision_recall_curve(data['label'], data['probability'])
    auprc = auc(recall, precision)

    # roc curve
    fpr, tpr, roc_thresholds = roc_curve(data['label'], data['probability'])

    # optimal threshold based on fbeta score
    best_fbeta = 0.0
    best_threshold = 0.0
    for threshold in prc_thresholds:
        preds = (data['probability'] >= threshold).astype(int)
        fbeta = fbeta_score(data['label'], preds, beta=0.5, zero_division=0)
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold

    preds = (data['probability'] >= best_threshold).astype(int)

    return {
        'fn': os.path.basename(file_path),

        # area metrics
        'auroc': roc_auc_score(data['label'], data['probability']),
        'auprc': auprc,
    
        # fields determined by chosen threshold
        'threshold': best_threshold,
        'f1_score': fbeta_score(data['label'], preds, beta=1, zero_division=0),
        'f0.5_score': fbeta_score(data['label'], preds, beta=0.5, zero_division=0),
        'accuracy': accuracy_score(data['label'], preds),
        'precision': precision_score(data['label'], preds, zero_division=0),
        'recall': recall_score(data['label'], preds, zero_division=0),

        # fields related to the ROC curve
        'roc_fpr': list(fpr),
        'roc_tpr': list(tpr),

        # fields related to precision recall curve
        'prc_precision': list(precision),
        'prc_recall': list(recall),
    }

# compute the average roc curve representing the approach as a whole
def average_curves(results):

    # common set of fpr/recall values
    common = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(common)
    mean_precision = np.zeros_like(common)

    for r in results:
        mean_tpr += np.interp(common, r['roc_fpr'], r['roc_tpr'])
        recall = list(reversed(r['prc_recall']))
        precision = list(reversed(r['prc_precision']))
        mean_precision += np.interp(common, recall, precision, left=None)
        # print(r['prc_recall'])
        # print(r['prc_precision'])
        # print(mean_precision)
        

    mean_tpr /= len(results)
    mean_precision /= len(results)

    mean_tpr[0] = 0
    mean_precision[0] = 1

    return {
        # average roc curve
        'roc_fpr': common,
        'roc_tpr': mean_tpr,
        'auroc': np.trapz(mean_tpr, common),
        
        # average pr curve
        'prc_precision': mean_precision,
        'prc_recall': common,
        'auprc': np.trapz(mean_precision, common),
    }


# iterate through all investigations, compute their metrics and return their average
def read_results(folder, max_workers=64):

    results = []
    files = [os.path.join(folder, fn) for fn in os.listdir(folder) if fn.endswith('csv')]

    # process asynchronously
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(get_result, fn): fn for fn in files}

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_file), total=len(files), desc="processing files"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing {future_to_file[future]}: {e}")

    return pd.DataFrame(results)

def generate_result_files():
    folders = [f for f in os.listdir(SAVE_PATH) if os.path.isdir(os.path.join(SAVE_PATH, f))]
    for folder in reversed(folders):
        result_file = os.path.join(SAVE_PATH, f'{folder}.csv')
        if not os.path.exists(result_file):
            result = read_results(os.path.join(SAVE_PATH, folder))
            result.to_csv(result_file, index=False)

def generate_metrics(name, n_tests=3):
    folders = [f'{name}_{i}' for i in range(1,n_tests+1)]
    result = {
        'auroc': [],
        'auprc': [],
        'f1_score': [],
        'f0.5_score': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
    }

    curves = []
    for folder in folders:
        result_file = os.path.join(SAVE_PATH, f'{folder}.csv')
        if not os.path.exists(result_file):
            raise ValueError('result files missing')
        df = pd.read_csv(result_file)
        for key in result:
            result[key].append(df[key].mean())

        curve_fields = ['roc_fpr', 'roc_tpr', 'prc_recall', 'prc_precision']
        for key in curve_fields:
            df[key] = df[key].apply(ast.literal_eval)
        
        rows = df[curve_fields].to_dict(orient='records')
        curves.extend(rows)
    
    curves = average_curves(curves)

    for key in result:
        mean = round(statistics.mean(result[key]) * 100, 2)
        stdev = round(statistics.stdev(result[key]) * 100, 2) if len(result[key]) > 1 else 0.0
        result[key] = (mean, stdev)
    print(result)
    
    # save curves
    pd.DataFrame({
        'fpr': curves['roc_fpr'],
        'tpr': curves['roc_tpr']
    }).to_csv(f'{SAVE_PATH}{name}_roc_curve.csv', index=False)

    pd.DataFrame({
        'recall': curves['prc_recall'],
        'precision': curves['prc_precision']
    }).to_csv(f'{SAVE_PATH}{name}_prc_curve.csv', index=False)

def generate_pca_visualisation(fn):
    df = pd.read_csv(fn)
    embeddings = df['embeds'].apply(ast.literal_eval).tolist()
    labels = df['label'].values
    X = np.array(embeddings)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    pca_df = pd.DataFrame(X_pca, columns=['pc1', 'pc2'])
    pca_df['label'] = labels

    # Create the plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='pc1', y='pc2', hue='label', palette='viridis', alpha=0.4)
    plt.title('PCA Visualization of Embeddings')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.legend(title='Class')
    plt.show()


if __name__ == '__main__':

    tests = [
        ('random_baseline', 3),
        ('majority_baseline', 1),
        ('optimal_baseline', 1),
        ('roberta_baseline', 3),
        ('dual_baseline', 3),
        ('dual_reptile', 1),
    ]
    for name, n_tests in tests:
        generate_metrics(name, n_tests=n_tests)
    # generate_result_files()
    
    # fn = 'Category_Wikipedia_sockpuppets_of_Cassandra872.csv'
    # fn = 'Category_Wikipedia_sockpuppets_of_Griot.csv'
    # folders = ['dual_reptile_1', 'dual_baseline_1']
    # for f in folders:
    #     fn2 = f'{SAVE_PATH}{f}/{fn}'
    #     generate_pca_visualisation(fn2)


