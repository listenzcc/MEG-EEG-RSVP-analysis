"""
File: decoding-sliding-auc-result.py
Author: Chuncheng Zhang
Date: 2026-04-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Get results for the decoding-sliding-auc.py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-03 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path(f'output/decoding-sliding-auc/')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

OUTPUT_DIR = DATA_DIR

# %% ---- 2026-04-03 ------------------------
# Function and class


# %% ---- 2026-04-03 ------------------------
# Play ground
folders = sorted(DATA_DIR.glob('*EG-S*'))
print(folders)
scores = np.concatenate(
    [np.load(e / 'scores.npy')[np.newaxis, :] for e in folders])
print(scores.shape)

times = np.linspace(-0.5, 1.5, 401)
print(f'{times.shape=}, {scores.shape=}')

# 转换为长格式 DataFrame
df_list = []
for i in range(scores.shape[0]):
    temp_df = pd.DataFrame({
        'time': times,
        'score': scores[i],
        'mode': folders[i].name.split('-')[0],
        'subj': folders[i].name.split('-')[1],
    })
    df_list.append(temp_df)

df_long = pd.concat(df_list, ignore_index=True)
print(df_long.head())

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

ax = axes[0]
sns.lineplot(data=df_long, x='time', y='score', hue='mode', ax=ax)
ax.set_title('Sliding AUC Scores Over Time')
ax.grid(True)

ax = axes[1]
sns.lineplot(data=df_long.query('mode == "MEG"'),
             x='time', y='score', hue='subj', ax=ax)
ax.set_title('MEG Scores Over Time')
ax.grid(True)

ax = axes[2]
sns.lineplot(data=df_long.query('mode == "EEG"'),
             x='time', y='score', hue='subj', ax=ax)
ax.set_title('EEG Scores Over Time')
ax.grid(True)

fig.tight_layout()

fname = OUTPUT_DIR / 'sliding_auc_scores.png'
fig.savefig(fname, dpi=400)
print(f'Saved figure to {fname}')

# plt.show()

# %% ---- 2026-04-03 ------------------------
# Pending

# %% ---- 2026-04-03 ------------------------
# Pending
