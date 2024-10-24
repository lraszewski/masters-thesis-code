import os
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from concurrent.futures import ThreadPoolExecutor, as_completed

# compute the metrics of a single investigation
def get_result(file_path):
    data = pd.read_csv(file_path)

    auroc = roc_auc_score(data['label'], data['probability'])

    precision, recall, _ = precision_recall_curve(data['label'], data['probability'])
    prauc = auc(recall, precision)

    return {
        'auroc': auroc,
        'prauc': prauc
    }

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

    # return mean
    results = pd.DataFrame(results)
    return results.mean()

if __name__ == '__main__':
    folder = '/mnt/d/results/roberta_baseline_1'
    results = read_results(folder)
    print(results)