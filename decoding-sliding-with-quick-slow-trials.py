"""
File: decoding-quick-slow-trials.py
Author: Chuncheng Zhang
Date: 2026-06-09
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding quick or slow target trials against non-target trials.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-09 ------------------------
# Requirements and constants
import json
import time
import argparse
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from util.easy_imports import *

# %%
# Args
MODE = 'EEG'
SUBJ = 'S02'
COND = 'quick'

try:
    assert __IPYTHON__, 'Using ipython'
    args = argparse.Namespace(mode=MODE, subj=SUBJ, context='IPython')

except:
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', help='Mode in EEG, MEG', default=MODE)
    parser.add_argument(
        '-s', '--subj', help='Subject in S01, S02, ...', default=SUBJ)
    parser.add_argument('-c', '--cond', help='Condition in quick, slow', default=COND)
    args = parser.parse_args()

print(f'{args=}')

# %%
STEP1_DIR = Path('./output/step-1')

STEP1_AVE_DIR = Path(f'output/step-1-subjects-average/{MODE}')

OUTPUT_DIR = Path('./output/decoding-sliding-with-quick-slow-trials/', f'{args.mode}-{args.subj}-{args.cond}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-06-09 ------------------------
# Function and class


# %% ---- 2026-06-09 ------------------------
# Play ground
epochs = mne.read_epochs(
    STEP1_DIR / f'{args.mode}-{args.subj}' / 'epochs-1-epo.fif')
epochs_nt = mne.read_epochs(
    STEP1_DIR / f'{args.mode}-{args.subj}' / 'epochs-2-epo.fif')

epochs

# %%
evoked2 = mne.read_evokeds(STEP1_AVE_DIR / '2-epo-ave.fif')[0]
evoked3 = mne.read_evokeds(STEP1_AVE_DIR / '3-epo-ave.fif')[0]

if args.mode == 'EEG':
    kwargs = dict(n_grad=0, n_mag=0, n_eeg=1)
elif args.mode == 'MEG':
    kwargs = dict(n_grad=0, n_mag=3, n_eeg=0)

proj2 = mne.compute_proj_evoked(evoked2, **kwargs)
proj3 = mne.compute_proj_evoked(evoked3, **kwargs)
proj = proj2 + proj3

epochs.add_proj(proj)
epochs.apply_proj()

epochs_nt.add_proj(proj)
epochs_nt.apply_proj()

# %%
df = pd.read_csv('output/diff-times-target.csv')
table_target = df.query(f'mode=="{args.mode}"').query(
    f'subject=="{args.subj}"').query('delay<1.0')
table_target

# %%
threshold = 0.4
epochs_quick = epochs[table_target.query(f'delay<{threshold}')['index']]
epochs_slow = epochs[table_target.query(f'delay>={threshold}')['index']]
print(f'{epochs_quick=}, {epochs_slow=}')


# %%
# ! epochs_t: epochs of targets.
# ! epochs_nt: epochs of non-targets.

epochs_t = epochs_quick if args.cond == 'quick' else epochs_slow
n1 = len(epochs_t)
n2 = len(epochs_nt)

# X shape is (n_epochs, n_channels, n_times)
X1 = epochs_t.get_data()
X2 = epochs_nt.get_data()
X = np.concatenate([X1, X2])

y = np.concatenate([
    np.zeros(n1) + 1,
    np.zeros(n2),
])

print(f'{X.shape=}, {y.shape=}')

# %%
clf = make_pipeline(
    Vectorizer(),          # (n_channels, n_times) → (features)
    StandardScaler(),
    SVC(kernel='rbf', probability=True)
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

time.sleep(1)

print(f'{scores.shape=}')  # (n_splits, n_times)
print(f'{scores=}')
print(f'{scores_mean=}')
# plt.plot(epochs.times, scores_mean)
# plt.show()


results = {
    'mode': args.mode,
    'subj': args.subj,
    'scores_mean': scores_mean.tolist(),
    'times': epochs.times.tolist()
}

dst = OUTPUT_DIR / 'decoding-sliding.json'
json.dump(results, open(dst, 'w'))


# %% ---- 2026-06-09 ------------------------
# Pending


# %% ---- 2026-06-09 ------------------------
# Pending
