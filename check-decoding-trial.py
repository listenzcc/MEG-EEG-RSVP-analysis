"""
File: check-decoding-trial.py
Author: Chuncheng Zhang
Date: 2026-05-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Check the results of decoding-trial[-124].py

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-05-06 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/decoding-trial-124')
assert DATA_DIR.is_dir(), 'Not found DATA_DIR'

# %% ---- 2026-05-06 ------------------------
# Function and class


# %% ---- 2026-05-06 ------------------------
# Play ground
files = sorted(DATA_DIR.rglob('*.json'))
print(files)
df = []
for p in tqdm(files):
    mode, subj = p.parent.name.split('-', 2)
    obj = json.load(open(p))
    report = obj['classification_report']['macro avg']
    report.update({
        'mode': mode,
        'subj': subj
    })
    df.append(report)
table = pd.DataFrame(df)
print(table.head())

# %%
# sns.barplot(table, x='mode', y='f1-score', hue='mode')
sns.violinplot(table, x='mode', y='f1-score', hue='mode')
plt.show()

# %% ---- 2026-05-06 ------------------------
# Pending


# %% ---- 2026-05-06 ------------------------
# Pending
