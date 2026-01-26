
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
MODE = 'MEG'

# %%


def read_raw(path: Path):
    raw = mne.io.read_raw_fif(path, preload=False)
    if MODE == 'MEG':
        raw.pick('mag')
    elif MODE == 'EEG':
        raw.pick([e for e in raw.ch_names if e not in ['CB1', 'CB2']])
    return raw


# %%
raw_folder = Path('data/RSVP_dataset/processed_data') / f'{MODE}_{SUBJ}'

block_files = list(raw_folder.glob('block_*_ica-raw.fif'))
block_files.sort()

raws = [read_raw(e) for e in block_files[:1]]
print(raws)

# %%
raw = iter(raws).__next__()

# Notch at 10 Hz
# raw.load_data()
# raw = raw.notch_filter(freqs=[10, 20, 30, 40], notch_widths=1, n_jobs=n_jobs)

events, event_id = mne.events_from_annotations(raw)
print(raw)
display(raw.info)
print(events.shape, event_id)

# %%
# Plot events
# mne.viz.plot_events(events, raw.info['sfreq'])
# plt.show()

# %%
tmin, tmax = -0.2, 1.0
l_freq, h_freq = 0.1, 40

epochs = mne.Epochs(raw, events, event_id, tmin=-0.2,
                    tmax=1.0, decim=int(raw.info['sfreq'] / 200))
epochs = epochs['1']
epochs.load_data()
epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

print(epochs)
evoked = epochs.average()
evoked.plot_joint()
plt.show()

# %%

# %% ---- 2026-01-26 ------------------------
# Function and class


# %% ---- 2026-01-26 ------------------------
# Play ground


# %% ---- 2026-01-26 ------------------------
# Pending


# %% ---- 2026-01-26 ------------------------
# Pending
