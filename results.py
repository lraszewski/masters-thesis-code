import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from concurrent.futures import ThreadPoolExecutor, as_completed

SAVE_PATH = '/mnt/d/results/'

# compute the metrics of a single investigation
def get_result(file_path):
    data = pd.read_csv(file_path)

    # auroc
    auroc = roc_auc_score(data['label'], data['probability'])

    # prauc
    precision, recall, _ = precision_recall_curve(data['label'], data['probability'])
    prauc = auc(recall, precision)

    # roc curve
    fpr, tpr, thresholds = roc_curve(data['label'], data['probability'])

    return {
        'auroc': auroc,
        'prauc': prauc,
        'fpr': fpr,
        'tpr': tpr
    }

# compute the average roc curve representing the approach as a whole
def average_roc_curve(results):

    # common set of fpr values
    common_fpr = np.linspace(0, 1, 100)
    mean_tpr = np.zeros_like(common_fpr)

    for r in results:
        mean_tpr += np.interp(common_fpr, r['fpr'], r['tpr'])
    
    mean_tpr /= len(results)
    average_auc = np.trapz(mean_tpr, common_fpr)

    return common_fpr, mean_tpr, average_auc


# iterate through all investigations, compute their metrics and return their average
def read_results(folder, max_workers=8):

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

    # get average roc curve
    fpr, tpr, auroc = average_roc_curve(results)
    roc_curve = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr
    })

    return pd.DataFrame(results), roc_curve

if __name__ == '__main__':
    name = 'dual_baseline'
    folder = SAVE_PATH + name
    results, roc_curve = read_results(folder)
    roc_curve.to_csv(SAVE_PATH + name + '_roc_curve.csv', index=False)

    print(results['auroc'].mean())
    print(results['prauc'].mean())