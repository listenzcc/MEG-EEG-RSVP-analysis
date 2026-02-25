"""
File: beta.tfr.evoked.py
Author: Chuncheng Zhang
Date: 2026-02-10
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compute TFR for evoked.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-10 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
MODE = 'MEG'

# %%
DATA_DIR = Path('output/step-1-average-all-subjects')

# %% ---- 2026-02-10 ------------------------
# Function and class


# %% ---- 2026-02-10 ------------------------
# Play ground
evoked = mne.read_evokeds(DATA_DIR / f'{MODE}-2-ave.fif')[0]

if MODE == 'EEG':
    evoked.pick(['O1', 'OZ', 'O2'])

if MODE == 'MEG':
    evoked.pick(['MLO52-4503', 'MZO03-4503', 'MRO52-4503'])


# %%
freqs = np.arange(2, 41, 2)
tfr = mne.time_frequency.tfr_morlet(
    evoked, freqs, n_cycles=2, n_jobs=n_jobs, return_itc=False)
display(tfr)

n = len(tfr.ch_names)

fig = evoked.plot_joint()

fig = evoked.compute_psd(fmax=freqs[-1]).plot()

fig, axes = plt.subplots(1, len(tfr.ch_names)+1,
                         figsize=(n*4+1, 4),
                         gridspec_kw={'width_ratios': [1]*n + [0.05]})

vmin, vmax = np.min(tfr.data), np.max(tfr.data)

# exclude last axis for colorbar
for ch, m, ax in zip(tfr.ch_names, tfr.data, axes[:-1]):
    im = ax.imshow(
        m,
        aspect='auto',
        extent=[tfr.times[0], tfr.times[-1], freqs[0], freqs[-1]],
        origin='lower',
        vmin=vmin, vmax=vmax,
        norm='log'  # log scale
    )
    ax.set_title(f'{ch}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

# put the colorbar to the largest col
cbar_ax = axes[-1]
plt.colorbar(im, cax=cbar_ax, label='Power (dB)')
plt.tight_layout()
plt.show()

# %%

# %% ---- 2026-02-10 ------------------------
# Pending


# %% ---- 2026-02-10 ------------------------
# Pending
