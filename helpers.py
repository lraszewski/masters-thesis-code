import pandas as pd
import os
from tqdm import tqdm

def read_investigation(fn):
    data = pd.read_csv(fn, dtype={'page': str, 'message': str}, lineterminator='\n', parse_dates=['timestamp'], date_format="%Y-%m-%dT%H:%M:%S%z")
    data['message'] = data['message'].fillna('')
    data['page'] = data['page'].fillna('')
    return data

def stats(fn, directory):
    data = read_investigation(os.path.join(directory, fn))

    pos_samples = data[(data['sock'] == 1)]
    neg_samples = data[(data['sock'] == 0)]

    puppetmaster = pos_samples['user'].mode()[0]
    puppetmaster_samples = data[data['user'] == puppetmaster]

    return {
        'name': fn,
        'length': data.shape[0],
        'duration': (pos_samples['timestamp'].max() - pos_samples['timestamp'].min()).total_seconds(),

        'num_positives': pos_samples.shape[0],
        'num_negatives': neg_samples.shape[0],
        'num_puppetmaster': puppetmaster_samples.shape[0],
        'num_sockpuppets': (pos_samples.shape[0] - puppetmaster_samples.shape[0]),

        'min_message_length': data['message'].apply(len).min(),
        'ave_message_length': data['message'].apply(len).mean(),
        'max_message_length': data['message'].apply(len).max(),

        'max_page_length': data['page'].apply(len).max(),
        'ave_page_length': data['page'].apply(len).mean(),
        'min_page_length': data['page'].apply(len).min(),
    }

def generate_stats(directory):
    data = [stats(fn, directory) for fn in tqdm(os.listdir(directory), desc='generating stats.csv') if fn.endswith('csv')]
    df = pd.DataFrame(data)
    if not os.path.exists('stats'):
        os.makedirs('stats')
    df.to_csv('stats/stats.csv', index=False)