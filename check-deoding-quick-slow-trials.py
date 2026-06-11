"""
File: check-deoding-quick-slow-trials.py
Author: Chuncheng Zhang
Date: 2026-06-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the results of decoding-quick-slow-trials.py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-09 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/decoding-quick-slow-trials')

# %% ---- 2026-06-09 ------------------------
# Function and class


# %% ---- 2026-06-09 ------------------------
# Play ground
array = []

for mode in ['MEG', 'EEG']:
    folder = DATA_DIR / mode
    for sub_folder in sorted(folder.iterdir()):
        subj = sub_folder.name
        print(mode, subj)
        src = sub_folder / 'decoding-sliding.json'
        if not src.is_file():
            continue
        obj = json.load(open(src))
        obj.update({'subj': subj, 'mode': mode})
        for t, s in zip(obj['times'], obj['scores_mean']):
            dct = {'time': t, 'score': s, 'subj': subj, 'mode': mode}
            array.append(dct)

df = pd.DataFrame(array)
df


# %% ---- 2026-06-09 ------------------------
# Pending
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.lineplot(df, x='time', y='score', hue='mode', ax=ax)
ax.axvline(0.4, color='gray')
ax.grid()
fig.savefig(DATA_DIR / 'decoding-quick-slow-trials.png')
plt.show()


# %% ---- 2026-06-09 ------------------------
# Pending
