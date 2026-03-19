"""
File: source-estimation.py
Author: Chuncheng Zhang
Date: 2026-03-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for evoked.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-19 ------------------------
# Requirements and constants
from util.easy_imports import *

from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator
from mne.datasets import fetch_fsaverage

# %%
DATA_DIR1 = Path(f'output/step-1-subjects-average-proj')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist.'

# %% ---- 2026-03-19 ------------------------
# Function and class


# %% ---- 2026-03-19 ------------------------
# Play ground

evt = '1'
for i, mode in enumerate(['MEG', 'EEG']):
    evoked = mne.read_evokeds(
        DATA_DIR1 / mode / f'{evt}-withproj-epo-ave.fif')[0]
    print(evoked)

# %% ---- 2026-03-19 ------------------------
# Pending


# %% ---- 2026-03-19 ------------------------
# Pending
