"""
File: decoding-sliding-auc-remove-artificial.py
Author: Chuncheng Zhang
Date: 2026-03-31
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding for the RSVP data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-31 ------------------------
# Requirements and constants
import matplotlib.pyplot as plt
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

if len(sys.argv) > 1:
    _, _,  SUBJ, _, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

DATA_DIR1 = Path(f'output/step-1-subjects-average/{MODE}')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist'

# %%
OUTPUT_DIR = Path(f'output/decoding-sliding-auc-remove-artificial/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%


# %% ---- 2026-03-31 ------------------------
# Function and class


# %% ---- 2026-03-31 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
# keypress
# epochs3 = mne.read_epochs(DATA_DIR / 'epochs-3-epo.fif', preload=True)
# others
# epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)

# print(epochs1, epochs2, epochs3, epochs4)

# %% ---- 2026-03-31 ------------------------
# Pending

# 合并所有epochs数据
epochs_all = mne.concatenate_epochs([
    epochs1,
    epochs2,
])
print(epochs_all)

if MODE == 'EEG':
    kwargs = dict(n_grad=0, n_mag=0, n_eeg=1)
elif MODE == 'MEG':
    kwargs = dict(n_grad=0, n_mag=3, n_eeg=0)

evoked2 = mne.read_evokeds(DATA_DIR1 / '2-epo-ave.fif')[0]
evoked3 = mne.read_evokeds(DATA_DIR1 / '3-epo-ave.fif')[0]
proj2 = mne.compute_proj_evoked(evoked2, **kwargs)
proj3 = mne.compute_proj_evoked(evoked3, **kwargs)
proj = proj2 + proj3

epochs_all.add_proj(proj)
epochs_all.apply_proj()

# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4 ...

# 分类器（每个时间点都会用）
clf = make_pipeline(
    Vectorizer(),          # (n_channels, n_times) → (features)
    StandardScaler(),
    SVC(kernel='rbf')
)

time_decod = SlidingEstimator(
    clf,
    scoring='roc_auc',
    n_jobs=-1
)

# 10-fold CV
cv = StratifiedKFold(n_splits=10, shuffle=True)

scores = cross_val_multiscore(
    time_decod,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

# 平均 across folds
scores_mean = scores.mean(axis=0)

print(scores.shape)  # (n_splits, n_times)
print(scores)
print(scores_mean)

scores_path = OUTPUT_DIR / 'scores.npy'
np.save(scores_path, scores_mean)
logger.info(f'Saved scores to {scores_path}')
exit(0)

# %%


# %% ---- 2026-03-31 ------------------------
# Pending
