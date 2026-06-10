"""
File: play.py
Author: Chuncheng Zhang
Date: 2026-06-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""

# %% ---- ERA from impulse response ------------------------

import mne
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.linalg import svd


# ---------------------------------------------------------
# Load ERP
# ---------------------------------------------------------

DATA_DIR = Path('./output/step-1-subjects-average')

evoked = mne.read_evokeds(
    DATA_DIR / 'EEG/1-epo-ave.fif'
)[0]

h = evoked.data[33].copy()

# 去均值
h = h - h.mean()

# ---------------------------------------------------------
# Hankel matrix
# ---------------------------------------------------------

# s = 50
# s = 200
s = 200

H0 = np.zeros((s, s))
H1 = np.zeros((s, s))

for i in range(s):
    for j in range(s):
        H0[i, j] = h[i + j]
        H1[i, j] = h[i + j + 1]

# ---------------------------------------------------------
# SVD
# ---------------------------------------------------------

U, S, Vt = svd(H0, full_matrices=False)

plt.figure(figsize=(6, 4))
plt.semilogy(S, 'o-')
plt.title('Singular Values')
plt.show()

# ---------------------------------------------------------
# choose order
# ---------------------------------------------------------

# r = 8
r = 16

Ur = U[:, :r]
Sr = np.diag(S[:r])
Vr = Vt[:r, :]

# ---------------------------------------------------------
# ERA realization
# ---------------------------------------------------------

Sr_sqrt = np.sqrt(Sr)
Sr_inv_sqrt = np.linalg.inv(Sr_sqrt)

A = (
    Sr_inv_sqrt
    @ Ur.T
    @ H1
    @ Vr.T
    @ Sr_inv_sqrt
)

B = Sr_sqrt @ Vr[:, [0]]

C = Ur[[0], :] @ Sr_sqrt

D = np.array([[h[0]]])

# ---------------------------------------------------------
# reconstruct impulse response
# ---------------------------------------------------------

N = len(h)

x = np.zeros((r, 1))

yhat = np.zeros(N)

for k in range(N):

    yhat[k] = (C @ x + D)[0, 0]

    if k == 0:
        u = 1.0
    else:
        u = 0.0

    x = A @ x + B * u

# ---------------------------------------------------------
# plot
# ---------------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(h, lw=3, label='ERP')
plt.plot(yhat, '--', lw=2, label='ERA')

plt.legend()
plt.show()

# ---------------------------------------------------------
# poles
# ---------------------------------------------------------

poles = np.linalg.eigvals(A)

plt.figure(figsize=(5, 5))

plt.scatter(
    poles.real,
    poles.imag
)

plt.axhline(0)
plt.axvline(0)

plt.xlabel('Real')
plt.ylabel('Imag')

plt.axis('equal')

plt.show()

print("Poles")
print(poles)

# %%
print(f'{A.shape=}, {B.shape=}, {C.shape=}, {D.shape=}')
# %%
