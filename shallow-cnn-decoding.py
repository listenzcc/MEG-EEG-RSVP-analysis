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
import numpy as np
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

# %% ---- 2026-06-03 ------------------------
# Function and class


# %% ---- 2026-06-03 ------------------------
# Play ground


# %% ---- 2026-06-03 ------------------------
# Pending


# ============================================================
# Example RSVP data
# ============================================================

# X:
# (trials, channels, time)

X = np.random.randn(1200, 64, 250).astype(np.float32)

# y:
# 0 = non-target
# 1 = target

y = np.random.randint(0, 2, 1200)


# ============================================================
# Tensor
# ============================================================

X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).long()

dataset = TensorDataset(X_tensor, y_tensor)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# ============================================================
# Model
# ============================================================

model = ShallowFBCSPNet(
    n_chans=64,
    n_outputs=2,
    n_times=250,

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

epochs = 20

model.train()

for epoch in range(epochs):

    total_loss = 0

    for batch_x, batch_y in loader:

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(logits, batch_y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(epoch, total_loss)


# ============================================================
# Inference
# ============================================================

model.eval()

with torch.no_grad():

    X_test = torch.tensor(X[:10]).to(device)

    logits = model(X_test)

    prob = F.softmax(logits, dim=1)

    target_prob = prob[:, 1]

print(target_prob)

# %% ---- 2026-06-03 ------------------------
# Pending
