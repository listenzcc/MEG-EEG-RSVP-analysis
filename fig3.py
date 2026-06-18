'''
Sliding decoding
'''

# %%
from statsmodels.stats.multitest import multipletests
from scipy import stats
from util.easy_imports import *

# %%
TIMES = np.linspace(-0.5, 1.5, 401)

# %%


def t_test_single(group, zero_value=0):
    t_stat, p = stats.ttest_1samp(group['score'] - zero_value, 0)
    # 单侧 p 值
    if t_stat > 0:
        p_one = p / 2
    else:
        p_one = 1 - p / 2
    return pd.Series({
        't_stat': t_stat,
        'p_value': p_one,
        'mean_score': group['score'].mean(),
        'n': len(group)
    })


# %%
'''
Decoding target
'''
# Read data
array = []

for data_dir, processing in zip([
    Path('./output/decoding-sliding-auc/'),
    Path('./output/decoding-sliding-auc-remove-artificial/'),
],
    ['raw', 'proj']
):

    scores_files = sorted(data_dir.rglob('*EG-S*/scores.npy'))

    for path in scores_files:
        mode, subj = path.parent.name.split('-')
        sliding_score = np.load(path)
        df = pd.DataFrame({
            'time': TIMES,
            'score': sliding_score,
            'mode': mode,
            'subj': subj,
            'processing': processing
        })
        array.append(df)

table = pd.concat(array)
table = table[table['time'] < 1.4]
# table = table[table['time'] > -0.4]
print(table)

# %%
# Statistic analysis

df = table.copy()

mean_score = df['score'].mean()
results = df.groupby(['time', 'mode', 'processing']
                     ).apply(lambda se: t_test_single(se, mean_score)).reset_index()

# 用上面的 results_df
results['p_adjusted'] = multipletests(
    results['p_value'], method='fdr_bh')[1]

# 查看校正后仍然显著的
significant_fdr = results[results['p_adjusted'] < 0.01].copy()
print(significant_fdr)

# %%
# Mark the significant value
significant_fdr['value'] = 0.0
significant_fdr.loc[significant_fdr.query(
    'mode=="MEG" & processing=="raw"').index, 'value'] = 1.2
significant_fdr.loc[significant_fdr.query(
    'mode=="MEG" & processing=="proj"').index, 'value'] = 1.15
significant_fdr.loc[significant_fdr.query(
    'mode=="EEG" & processing=="raw"').index, 'value'] = 1.10
significant_fdr.loc[significant_fdr.query(
    'mode=="EEG" & processing=="proj"').index, 'value'] = 1.05

# Find the first significant time points
group = significant_fdr.groupby(['mode', 'processing'])
significant_fdr_first_time = group.min(numeric_only=True).reset_index()

# %%
# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
sns.lineplot(df, x='time', y='score', hue='mode',
             style='processing', hue_order=['EEG', 'MEG'], ax=ax)
sns.scatterplot(significant_fdr, x='time', y='value', edgecolor='none',
                hue='mode', style='processing', hue_order=['EEG', 'MEG'], ax=ax)
plt.axvline(0, linestyle='--', color='gray')
plt.axvline(0.47, linestyle='--', color='gray')
# 在每个点旁边添加 time 标签
for i, row in significant_fdr_first_time.iterrows():
    plt.text(row['time'], row['value'], f'{row["time"]:.2f}',
             fontsize=8, ha='center', va='bottom')
plt.suptitle('AUC Sliding')
plt.savefig('collect-results/fig3/auc-sliding.png')
plt.show()

# %%
'''
Decoding target (quick vs slow)
'''
# Read data
data_dir = Path('./output/decoding-sliding-with-quick-slow-trials/')
found = sorted(data_dir.rglob('decoding-sliding.json'))

array = []

for path in found:
    mode, subj, cond = path.parent.name.split('-')
    obj = json.load(open(path))
    for t, s in zip(obj['times'], obj['scores_mean']):
        array.append({'time': t, 'score': s, 'mode': mode,
                     'subj': subj, 'cond': cond})
table = pd.DataFrame(array)
table = table[table['time'] < 1.4]
# table = table[table['time'] > -0.4]
print(table)

# %%
# Statistical analysis
df = table.copy()

mean_score = df['score'].mean()
results = df.groupby(['time', 'mode', 'cond']
                     ).apply(lambda se: t_test_single(se, mean_score)).reset_index()

# 用上面的 results_df
results['p_adjusted'] = multipletests(
    results['p_value'], method='fdr_bh')[1]

# 查看校正后仍然显著的
significant_fdr = results[results['p_adjusted'] < 0.01].copy()
print(significant_fdr)

# %%
# Mark the significant value
significant_fdr['value'] = 0.0
significant_fdr.loc[significant_fdr.query(
    'mode=="MEG" & cond=="quick"').index, 'value'] = 1.1
significant_fdr.loc[significant_fdr.query(
    'mode=="MEG" & cond=="slow"').index, 'value'] = 1.0
significant_fdr.loc[significant_fdr.query(
    'mode=="EEG" & cond=="quick"').index, 'value'] = 1.1
significant_fdr.loc[significant_fdr.query(
    'mode=="EEG" & cond=="slow"').index, 'value'] = 1.0

# Find the first significant time points
group = significant_fdr.groupby(['mode', 'cond'])
significant_fdr_first_time = group.min(numeric_only=True).reset_index()

# %%
# Plot

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, mode in zip(axes, ['MEG', 'EEG']):
    _df = df.query(f'mode=="{mode}"')
    _dfm = significant_fdr.query(f'mode=="{mode}"')
    _dfmft = significant_fdr_first_time.query(f'mode=="{mode}"')

    sns.lineplot(_df, x='time', y='score', hue='cond',
                 hue_order=['quick', 'slow'], ax=ax)
    ax.axvline(0, linestyle='--', color='gray')
    ax.axvline(0.4, linestyle='--', color='gray')

    # if mode == 'MEG':
    #     ax.axvline(0.38, linestyle='--', color='gray')
    #     ax.axvline(0.46, linestyle='--', color='gray')

    sns.scatterplot(_dfm, x='time', y='value', edgecolor='none',
                    hue='cond', hue_order=['quick', 'slow'], ax=ax)
    # # 在每个点旁边添加 time 标签
    for i, row in _dfmft.iterrows():
        ax.text(row['time'], row['value'], f'{row["time"]:.2f}',
                fontsize=8, ha='center', va='bottom')

    ax.set_title(f'{mode}')

plt.suptitle('AUC Sliding (quick vs slow)')
plt.savefig('collect-results/fig3/auc-sliding-quick-vs-slow.png')
plt.show()

# %%
'''
Keypress delay
'''
table = pd.read_csv('./output/diff-times-keypress.csv')
print(table)

df = table.copy()
df = df[df['delay'] < 1]  # Filter out delays greater than 1 second
sns.histplot(df, x='delay', hue='mode', element='step',
             stat='density', legend=True)
plt.axvline(0.4, linestyle='--', color='gray')
plt.xlim((0.2, 0.6))
plt.xlabel('Delay (s)')
plt.title('Distribution of Delays between Target and Keypress Events')
plt.savefig('./collect-results/fig3/diff-times-keypress.png')
plt.show()

# %%
