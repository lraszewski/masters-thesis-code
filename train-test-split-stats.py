import pandas as pd

df = pd.read_csv('stats/stats.csv')

# filter to actual task distribution
min_puppetmaster = 10
min_sockpuppets = 5
min_ratio = 1
df = df[df['num_puppetmaster'] >= min_puppetmaster]
df = df[df['num_sockpuppets'] >= min_sockpuppets]
df = df[df['num_negatives'] >= df['num_positives'] * min_ratio]
print(len(df))

df['ratio'] = df['num_puppetmaster'] / df['num_positives']
df['support_negatives'] = (df['num_negatives'] * df['ratio']).round().astype(int)
df['test_negatives'] = df['num_negatives'] - df['support_negatives']

val_ratio = 0.8
df['train_negatives'] = (df['support_negatives'] * val_ratio).round().astype(int)
df['train_positives'] = (df['num_puppetmaster'] * val_ratio).round().astype(int)
df['val_negatives'] = df['support_negatives'] - df['train_negatives']
df['val_positives'] = df['num_puppetmaster'] - df['train_positives']

result = {
    'ave_train_pos': df['train_positives'].mean(),
    'ave_train_neg': df['train_negatives'].mean(),
    'ave_val_pos': df['val_positives'].mean(),
    'ave_val_neg': df['val_negatives'].mean(),
    'ave_test_pos': df['num_sockpuppets'].mean(),
    'ave_test_neg': df['test_negatives'].mean()
}

result['ave_train'] = result['ave_train_pos'] + result['ave_train_neg']
result['ave_val'] = result['ave_val_pos'] + result['ave_val_neg']
result['ave_test'] = result['ave_test_pos'] + result['ave_test_neg']

for k, v in result.items():
    print(f'{k}: {v:.2f}')
