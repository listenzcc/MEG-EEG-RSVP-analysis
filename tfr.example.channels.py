"""
File: tfr.example.channels.py
Author: Chuncheng Zhang
Date: 2026-02-28
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot signal on example channels.
    Compute tfr on example channels.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-28 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR1 = Path(f'output/step-1-subjects-average-proj')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist.'

DATA_DIR = Path('output/step-1-subjects-average')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist.'

# %% ---- 2026-02-28 ------------------------
# Function and class


def get_picks(mode):
    if mode == 'EEG':
        picks = ['O1', 'OZ', 'O2', 'C3', 'CZ', 'C4']
    else:
        picks = ['MLO52', 'MZO03', 'MRO52', 'MLC42', 'MZC03', 'MRC42']
    return picks


# %% ---- 2026-02-28 ------------------------
# Play ground

fig_eeg, axes_eeg = plt.subplots(4, 6, figsize=(16, 8))
fig_meg, axes_meg = plt.subplots(4, 6, figsize=(16, 8))

for mode, (i, evt) in product(['EEG', 'MEG'], enumerate(['1', '2', '3', '1p'])):
    print(mode, i, evt)
    picks = get_picks(mode)
    if evt == '1p':
        evoked = mne.read_evokeds(
            DATA_DIR1 / mode / '1-withproj-epo-ave.fif')[0]
    else:
        evoked = mne.read_evokeds(DATA_DIR / mode / f'{evt}-epo-ave.fif')[0]
    evoked.pick(picks)

    # Time frequency analysis
    # I want the n_cycles as large as possible
    freqs = np.arange(2, 31, 1)
    t = evoked.times[-1] - evoked.times[0]
    n_cycles = [int(f*t*0.5) for f in freqs]
    tfr = mne.time_frequency.tfr_morlet(
        evoked,
        freqs=freqs,
        n_cycles=n_cycles,
        n_jobs=n_jobs,
        return_itc=False)

    for j, ch_name in enumerate(tfr.ch_names):
        if mode == 'EEG':
            ax = axes_eeg[i, j]
        else:
            ax = axes_meg[i, j]

        # tfr.apply_baseline(mode='ratio', baseline=(None, 0), verbose=False)
        tfr.plot(picks=ch_name, axes=ax, show=False)
        ax.set_title(f'{ch_name} ({evt=})')

fig_eeg.tight_layout()
fig_meg.tight_layout()

fig_eeg.savefig('output/fig-eeg.png')
fig_meg.savefig('output/fig-meg.png')

plt.show()

# %%

# %% ---- 2026-02-28 ------------------------
# Pending


# %% ---- 2026-02-28 ------------------------
# Pending
