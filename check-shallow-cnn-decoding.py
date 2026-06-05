"""
File: check-shallow-cnn-decoding.py
Author: Chuncheng Zhang
Date: 2026-06-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the results of shallow-cnn-decoding.py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-04 ------------------------
# Requirements and constants
from sklearn import metrics
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/shallow-cnn-decoding-trial')
DATA_DIR = Path('./output/shallow-cnn-decoding-trial-124')


# %% ---- 2026-06-04 ------------------------
# Function and class


# %% ---- 2026-06-04 ------------------------
# Play ground
collected = []
for folder in [e for e in DATA_DIR.iterdir() if e.is_dir()]:
    mode, subj = folder.name.split('-')
    print(mode, subj)
    for p in folder.iterdir():
        obj = json.load(open(p))
        dct = obj['report']['macro avg']
        y_score = obj['y_prob']
        y_true = obj['y_true']
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_score)
        dct.update({
            'mode': mode,
            'subj': subj,
            'auc': auc
        })
        collected.append(dct)

table = pd.DataFrame(collected)
print(table)

# %%
sns.barplot(table, x='mode', y='f1-score', hue='mode')
plt.show()
sns.barplot(table, x='mode', y='auc', hue='mode')
plt.show()

# %%
group = table.groupby('mode')
group.mean(numeric_only=True)

# %% ---- 2026-06-04 ------------------------
# Pending

# %% ---- 2026-06-04 ------------------------
# Pending
