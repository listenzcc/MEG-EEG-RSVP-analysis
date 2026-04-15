"""
File: check-source-decoding-sliding.py
Author: Chuncheng Zhang
Date: 2026-04-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the source decoding sliding results.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-13 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/source-decoding-sliding-auc')
assert DATA_DIR.is_dir(), f'{DATA_DIR} does not exist'

OUTPUT_DIR = Path('./output/source-decoding-sliding-auc')

# %% ---- 2026-04-13 ------------------------
# Function and class


# %% ---- 2026-04-13 ------------------------
# Play ground
files = sorted(DATA_DIR.rglob('*/regions/*.json'))

df = []
for p in tqdm(files, 'parse file names'):
    mode, subj = p.relative_to(DATA_DIR).parts[0].split('-')
    # p.name is "scores_[name]_[lh, rh].json"
    area = p.name[7:-8]
    hemi = p.name[-7:-5]
    df.append({'mode': mode, 'subj': subj, 'hemi': hemi, 'area': area, 'p': p})
table = pd.DataFrame(df)
print(table)

# %%
table['peak'] = 0.0
table['delay'] = 0.0
times = np.linspace(-0.5, 1.5, 401)
for i, row in tqdm(table.iterrows(), 'find peaks', total=len(table)):
    # Read, data shape: (n_cv, n_times)
    data = np.array(json.load(open(row['p'])))
    # Average on cv, data shape -> (n_times,)
    data = np.mean(data, axis=0)
    peak = np.max(data)
    delay = times[np.argmax(data)]
    table.at[i, 'peak'] = peak
    table.at[i, 'delay'] = delay

print(table)

# %%
plt.clf()
sns.violinplot(table, x='mode', y='peak', hue='hemi')
plt.title('Peak AUC')
plt.savefig(OUTPUT_DIR / 'peak_auc.png')
# plt.show()

# %%
plt.clf()
sns.violinplot(table, x='mode', y='delay', hue='hemi')
plt.title('Peak AUC Delay')
plt.savefig(OUTPUT_DIR / 'peak_delay.png')
# plt.show()

# %%
df = table.groupby(['mode', 'subj', 'hemi'])['peak'].max().reset_index()

plt.clf()
sns.violinplot(df, x='mode', y='peak', hue='hemi')
plt.title('Peak AUC (max over areas)')
plt.savefig(OUTPUT_DIR / 'peak_auc_max.png')
# plt.show()

# %% ---- 2026-04-13 ------------------------
# Pending


# %% ---- 2026-04-13 ------------------------
# Pending
