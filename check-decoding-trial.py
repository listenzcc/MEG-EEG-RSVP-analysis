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
from sklearn.metrics import roc_auc_score
from util.easy_imports import *

# %%
DATA_DIR = Path('./output/decoding-trial-124')
assert DATA_DIR.is_dir(), 'Not found DATA_DIR'

# %% ---- 2026-05-06 ------------------------
# Function and class


def mk_table(data_dir: Path, condition: str = 'default'):
    files = sorted(data_dir.rglob('*.json'))
    df = []
    for p in tqdm(files):
        mode, subj = p.parent.name.split('-', 2)
        obj = json.load(open(p))
        report = obj['classification_report']['macro avg']
        pred_proba = np.array(obj['pred_proba'])[:, 0]
        y_true = np.array(obj['y_true']) == 1
        auc = roc_auc_score(y_true, pred_proba)
        acc = obj['accuracy']

        report.update({
            'mode': mode,
            'subj': subj,
            'auc': auc,
            'acc': acc
        })
        df.append(report)
    table = pd.DataFrame(df)
    table['condition'] = condition
    return table


# %% ---- 2026-05-06 ------------------------
# Play ground
# table = mk_table(DATA_DIR)
table = pd.concat([
    mk_table(Path('./output/decoding-trial'), '12'),
    mk_table(Path('./output/decoding-trial-124'), '124'),
])
table.reset_index()
print(table.head())

# %%
sns.boxplot(table, x='condition', hue='mode', y='acc')
plt.title('ACC')
plt.savefig('./output/decoding-acc.png')
plt.show()

sns.boxplot(table, x='condition', hue='mode', y='auc')
plt.title('AUC')
plt.savefig('./output/decoding-auc.png')
plt.show()

sns.boxplot(table, x='condition', hue='mode', y='f1-score')
plt.title('F1 score (macro avg)')
plt.savefig('./output/decoding-f1-score.png')
plt.show()


# %% ---- 2026-05-06 ------------------------
# Pending


# %% ---- 2026-05-06 ------------------------
# Pending
