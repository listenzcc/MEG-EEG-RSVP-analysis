"""
File: prepare-for-deeplearning-decoding.py
Author: Chuncheng Zhang
Date: 2026-04-24
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Prepare data for deeplearning decoding.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-24 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

# %%
OUTPUT_DIR = Path(f'output/decoding-using-deeplearning/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-04-24 ------------------------
# Function and class


# %% ---- 2026-04-24 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
# epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)
# keypress
# epochs3 = mne.read_epochs(DATA_DIR / 'epochs-3-epo.fif', preload=True)
# others
# epochs4 = mne.read_epochs(DATA_DIR / 'epochs-4-epo.fif', preload=True)

print(epochs1)
# print(epochs2)
# print(epochs3)
# print(epochs4)


# %%
epochs = epochs1.copy()
epochs.crop(0, 1)
print(epochs.info['sfreq'])
X = epochs.get_data()
print(X.shape)

# %% ---- 2026-04-24 ------------------------
# Pending


# %% ---- 2026-04-24 ------------------------
# Pending
