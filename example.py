
"""
File: example.py
Author: Chuncheng Zhang
Date: 2026-01-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read raw file for example.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-01-26 ------------------------
# Requirements and constants
from util.easy_imports import *
from util.summary_data import summarize_dataset

# print(summarize_dataset(Path("data/RSVP_dataset/processed_data")))

# %%
SUBJ = 'S02'
MODE = 'EEG'

# %%
raw_folder = Path('data/RSVP_dataset/processed_data') / f'{MODE}_{SUBJ}'

block_files = list(raw_folder.glob('block_*_ica-raw.fif'))
block_files.sort()

raws = [mne.io.read_raw_fif(e, preload=False) for e in block_files]
print(raws)

# %%
raw = iter(raws).__next__()
events, event_id = mne.events_from_annotations(raw)
print(raw)
display(raw.info)
print(events.shape, event_id)

# %%
mne.viz.plot_events(events)

# %%
epochs = mne.Epochs(raw, events, event_id, tmin=-0.2,
                    tmax=1.0, decim=int(raw.info['sfreq'] / 200))
print(epochs)

# %% ---- 2026-01-26 ------------------------
# Function and class


# %% ---- 2026-01-26 ------------------------
# Play ground


# %% ---- 2026-01-26 ------------------------
# Pending


# %% ---- 2026-01-26 ------------------------
# Pending
