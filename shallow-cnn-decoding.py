"""
File: shallow-cnn-decoding.py
Author: Chuncheng Zhang
Date: 2026-06-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding with shallow-cnn

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-03 ------------------------
# Requirements and constants
import time
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import argparse
import numpy as np
from sklearn.model_selection import StratifiedKFold
from util.easy_imports import *

import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from braindecode.models import ShallowFBCSPNet

# ============================================================
# Device
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

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
OUTPUT_DIR = Path(f'output/shallow-cnn-decoding-trial-124/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-06-03 ------------------------
# Function and class


# %% ---- 2026-06-03 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
# others
epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)

# %% ---- 2026-06-03 ------------------------
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
y[y > 1] = 0

print(f'{X.shape=}, {np.unique(y)=}')

# 10-fold CV
# cv = StratifiedKFold(n_splits=10, shuffle=True)
cv = StratifiedKFold(n_splits=10, shuffle=True)
print(cv)

# Use the first folder only.
for train_idx, test_idx in cv.split(X, y):
    break

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]


# Scale each channel independently across trials and time
scaler = StandardScaler()
n_trials, n_chans, n_times = X.shape
X_reshaped = X.reshape(-1, n_chans)          # (trials*time, channels)
X_scaled = scaler.fit_transform(X_reshaped).reshape(n_trials, n_chans, n_times)
X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]

print(f'{X_train.shape=}, {y_train.shape=}, {X_test.shape=}, {y_test.shape=}')

# %%
X_train_tensor = torch.tensor(X_train.astype(np.float32))
y_train_tensor = torch.tensor(y_train).long()
X_test_tensor = torch.tensor(X_test.astype(np.float32))
y_test_tensor = torch.tensor(y_test).long()

# exit(0)

# %%

# ============================================================
# Example RSVP data
# ============================================================

# X:
# (trials, channels, time)
# X = np.random.randn(1200, 64, 250).astype(np.float32)
# X = np.random.randn(1200, n_chans, n_times).astype(np.float32)

# y:
# 0 = non-target
# 1 = target
# y = np.random.randint(0, 2, 1200)
# y = np.random.randint(0, 2, 1200)


# ============================================================
# Tensor
# ============================================================

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).long()
# dataset = TensorDataset(X_tensor, y_tensor)
dataset = TensorDataset(X_train_tensor, y_train_tensor)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# ============================================================
# Model
# ============================================================

model = ShallowFBCSPNet(
    n_chans=n_chans,
    n_outputs=2,
    n_times=n_times,

    final_conv_length="auto",

    # RSVP建议
    filter_time_length=50,
    pool_time_length=75,
    pool_time_stride=15,

    drop_prob=0.5
)

model.to(device)


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

# ============================================================
# Train
# ============================================================

epochs = 50

model.train()

for epoch in tqdm(range(epochs)):

    total_loss = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # print(f'{batch_x.shape=}, {batch_y.shape=}')

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        total_loss += loss.item()

    print(epoch, total_loss)


# ============================================================
# Inference
# ============================================================

print('---- Evaluation ----')
model.eval()

with torch.no_grad():
    # X_test = torch.tensor(X[:10]).to(device)
    logits = model(X_test_tensor.to(device))
    prob = F.softmax(logits, dim=1)
    target_prob = prob[:, 1]

y_prob = target_prob.cpu().numpy()
y_pred = y_prob > 0.5
y_true = y_test

# print(y_prob, y_pred, y_true)

report = metrics.classification_report(
    y_true=y_true, y_pred=y_pred, output_dict=True)
print(report)

results = {
    'y_prob': y_prob.tolist(),
    'y_pred': y_pred.tolist(),
    'y_true': y_true.tolist(),
    'report': report
}

with open(OUTPUT_DIR / f'results-{time.time()}.json', 'w') as f:
    json.dump(results, f)
    print(f'Saved results into {f.name}')


# %% ---- 2026-06-03 ------------------------
# Pending
