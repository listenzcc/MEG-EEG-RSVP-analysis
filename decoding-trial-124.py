"""
File: decoding-trial.py
Author: Chuncheng Zhang
Date: 2026-04-29
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding trials.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-29 ------------------------
# Requirements and constants
from sklearn.metrics import accuracy_score, classification_report
import argparse
import matplotlib.pyplot as plt
from mne.decoding import Vectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

try:
    # In IPython environment, use the default values
    assert __IPYTHON__, 'Not in IPython environment, please provide arguments'
except:
    # In command line, parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--subj', default=SUBJ)
    parser.add_argument('--mode', default=MODE)
    args = parser.parse_args()

    SUBJ = args.subj
    MODE = args.mode

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

# %%
OUTPUT_DIR = Path(f'output/decoding-trial-124/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-04-29 ------------------------
# Function and class


# %% ---- 2026-04-29 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
# keypress
# epochs3 = mne.read_epochs(DATA_DIR / 'epochs-3-epo.fif', preload=True)
# others
epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)


# %% ---- 2026-04-29 ------------------------
# Pending
# 合并所有epochs数据
epochs_all = mne.concatenate_epochs([
    epochs1,
    epochs2,
    epochs4
])
print(epochs_all)

# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4 ...
y[y == 4] = 2

print(f'{X.shape=}, {np.unique(y)=}')

# %%

# 分类器（每个时间点都会用）
clf = make_pipeline(
    Vectorizer(),          # (n_channels, n_times) → (features)
    StandardScaler(),
    SVC(kernel='rbf', probability=True)
)

# 10-fold CV
# cv = StratifiedKFold(n_splits=10, shuffle=True)
cv = StratifiedKFold(n_splits=10, shuffle=True)

pred_proba = cross_val_predict(
    clf, X, y, cv=cv, method='predict_proba', n_jobs=-1)
y_pred = np.argmax(pred_proba, axis=1) + 1  # 因为标签从1开始，所以加1
print(f'{pred_proba=}')
print(f'{y_pred=}')
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)
print(f'Accuracy: {acc:.4f}')
print(f'Classification Report:\n{report}')
# scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
# print(f'{scores=}, {scores.mean()=:.4f}')

# Save results
results = {
    'accuracy': acc,
    'classification_report': report,
    'pred_proba': pred_proba,
    'y_pred': y_pred,
    'y_true': y
}
results = {k: v.tolist() if isinstance(v, np.ndarray)
           else v for k, v in results.items()}
fname = OUTPUT_DIR / 'results.json'
json.dump(results, open(fname, 'w'), indent=4)
logger.info(f'Saved results to {fname}')


# %% ---- 2026-04-29 ------------------------
# Pending
