"""
File: decoding-hdca.py
Author: Chuncheng Zhang
Date: 2026-05-18
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with hdca classifier.
    Compare with RBF-SVM, the HDCA's accuracy is not higher.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-05-18 ------------------------
# Requirements and constants
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from mne.decoding import Vectorizer
from sklearn.metrics import accuracy_score, classification_report
import argparse
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
# OUTPUT_DIR = Path(f'output/decoding-trial/{MODE}-{SUBJ}')
OUTPUT_DIR = Path('tmp')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-05-18 ------------------------
# Function and class


def hdca_features(X, y, time_windows, spatial_regularization=1e-4):
    """
    Extract HDCA features from EEG data.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_times)
        EEG data.
    y : ndarray, shape (n_trials,)
        Class labels (binary, e.g., 0 and 1).
    time_windows : list of (start, end) indices
        Each tuple defines a time window (in samples).
    spatial_regularization : float
        Regularisation term for the spatial filter covariance inversion.

    Returns
    -------
    features : ndarray, shape (n_trials, n_windows)
        Extracted features for each trial and window.
    filters : list of ndarray, shape (n_channels,) per window
        Spatial filters for each window.
    """
    n_trials, n_channels, n_times = X.shape
    n_windows = len(time_windows)
    features = np.zeros((n_trials, n_windows))
    filters = []

    for w_idx, (start, end) in enumerate(time_windows):
        # Data in this window: (trials, channels, window_length)
        X_win = X[:, :, start:end]

        # Compute class means and within-class covariance
        class_means = []
        covs = []
        for label in np.unique(y):
            X_class = X_win[y == label]
            class_means.append(np.mean(X_class, axis=(0, 2))
                               )  # mean over trials and time
            # covariance over channels (pooling time and trials)
            X_flat = X_class.reshape(X_class.shape[0], n_channels, -1)
            X_flat = X_flat.transpose(1, 0, 2).reshape(n_channels, -1)
            covs.append(np.cov(X_flat))

        # Pooled within-class covariance
        Sw = (covs[0] * (y == np.unique(y)[0]).sum() +
              covs[1] * (y == np.unique(y)[1]).sum()) / n_trials
        Sw += spatial_regularization * np.eye(n_channels)   # regularization

        # Difference of class means (discriminant vector)
        d = class_means[1] - class_means[0]

        # Spatial filter: w = inv(Sw) * d
        # (maximises SNR of the projected class means)
        w = np.linalg.solve(Sw, d)
        # Normalise filter to unit norm
        w = w / np.linalg.norm(w)
        filters.append(w)

        # Project each trial: first apply filter to each time point, then average
        # Equivalent to: feature = mean( w^T * X_win ) over time
        # For efficiency: compute dot product (w^T * X) per trial and average over time
        trial_features = np.tensordot(w, X_win, axes=(
            [0], [1]))  # shape (n_trials, win_len)
        features[:, w_idx] = np.mean(trial_features, axis=1)

    return features, filters


# %% ---- 2026-05-18 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
# keypress
# epochs3 = mne.read_epochs(DATA_DIR / 'epochs-3-epo.fif', preload=True)
# others
# epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)

# %%
# 合并所有epochs数据
epochs_all = mne.concatenate_epochs([
    epochs1,
    epochs2,
])
print(epochs_all)

# 获取数据X和标签y
X = epochs_all.get_data()  # shape: (n_epochs, n_channels, n_times)
y = epochs_all.events[:, -1]  # 标签 1,2,3,4 ...

print(f'{X.shape=}, {np.unique(y)=}')

# %% ---- 2026-05-18 ------------------------
# Pending
skf = StratifiedKFold(n_splits=10, shuffle=True)
# print(skf.get_n_splits(X, y))

# for train_idx, test_idx in skf.split(X, y):
#     print(f'{train_idx=}, {test_idx=}')
#     break

fs = 200  # Hz
n_times = 401
# Define time windows for HDCA (sliding, 50% overlap)
window_duration = 0.080  # 80 ms
step = 0.040             # 40 ms step (50% overlap)

# ! window_duration is the lower the better.
window_duration = 0.040  # 80 ms
step = window_duration * 0.5             # 40 ms step (50% overlap)

win_len = int(window_duration * fs)
step_samples = int(step * fs)
starts = range(0, n_times - win_len + 1, step_samples)
time_windows = [(s, s + win_len) for s in starts]
print(f"Number of windows: {len(time_windows)}")

# Extract HDCA features
features, filters = hdca_features(
    X, y, time_windows, spatial_regularization=1e-4)

print(f'{features.shape=}')

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

clf = SVC(kernel='rbf', probability=True)
scores = cross_val_score(clf, features_scaled, y, cv=skf, scoring='accuracy')
print(
    f"HDCA decoding accuracy: {np.mean(scores)*100:.1f}% ± {np.std(scores)*100:.1f}%")

pred_proba = cross_val_predict(
    clf, features_scaled, y, cv=skf, method='predict_proba', n_jobs=-1)
y_pred = np.argmax(pred_proba, axis=1) + 1  # 因为标签从1开始，所以加1
acc = accuracy_score(y, y_pred)
report = classification_report(y, y_pred, output_dict=True)
print(f'Accuracy: {acc:.4f}')
print(f'Classification Report:\n{report}')
# %% ---- 2026-05-18 ------------------------
# Pending
