"""
File: check-decoding-sliding-with-quick-slow-trials.py
Author: Chuncheng Zhang
Date: 2026-06-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the results of decoding-sliding-with-quick-slow-trials.py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-10 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/decoding-sliding-with-quick-slow-trials/')

# %% ---- 2026-06-10 ------------------------
# Function and class


# %% ---- 2026-06-10 ------------------------
# Play ground
found = sorted(DATA_DIR.rglob('decoding-sliding.json'))
# print(found)

array = []

for path in found:
    mode, subj, cond = path.parent.name.split('-')
    print(mode, subj, cond)
    obj = json.load(open(path))
    for t, s in zip(obj['times'], obj['scores_mean']):
        array.append({'time': t, 'score': s, 'mode': mode,
                     'subj': subj, 'cond': cond})
table = pd.DataFrame(array)
print(table.head())


# %% ---- 2026-06-10 ------------------------
# Pending
df = table.copy()
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.lineplot(df, x='time', y='score', hue='mode', style='cond', ax=ax)
ax.axvline(0.4, color='gray')
ax.grid()
fig.savefig(DATA_DIR / 'decoding-sliding-with-quick-slow-trials.png')
plt.show()

# %% ---- 2026-06-10 ------------------------
# Pending
