"""
File: 2.1.detect.erp.py
Author: Chuncheng Zhang
Date: 2026-02-02
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Detect the ERP from the RSVP data.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-02 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
SUBJ = 'S05'
MODE = 'MEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

DATA_DIR = Path('./output/step-1') / f'{MODE}-{SUBJ}'
OUTPUT_DIR = Path('./output/step-2') / f'{MODE}-{SUBJ}'
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-02-02 ------------------------
# Function and class


# %% ---- 2026-02-02 ------------------------
# Play ground

# Read data
epochs13 = mne.read_epochs(DATA_DIR / 'epochs-1-3-epo.fif')
evoked1 = mne.read_evokeds(DATA_DIR / 'evoked-1-ave.fif')[0]
evoked2 = mne.read_evokeds(DATA_DIR / 'evoked-2-ave.fif')[0]
evoked3 = mne.read_evokeds(DATA_DIR / 'evoked-3-ave.fif')[0]

print(epochs13)
print(evoked1)
print(evoked2)
print(evoked3)

# %%
# Project for keypress artificial

# Project using epochs
# e = epochs13['3'].copy()
# e.crop(0, 0.1)
# proj = mne.compute_proj_epochs(e, n_mag=3, n_eeg=3)

# Project using evoked
# ! It seems better
e = evoked3.copy()
e.crop(0, 0.1)
proj = mne.compute_proj_evoked(e, n_mag=3, n_eeg=3)
print(proj)

# %%
# Plot clear ERP
# Make ERP evoked and epochs
epochs = epochs13['1'].copy()

# Remove SSVEP
data = epochs.get_data()
for d in data:
    d -= evoked2.data
epochs = mne.EpochsArray(data, epochs.info, events=epochs.events,
                         event_id=epochs.event_id, tmin=epochs.tmin)
# Project key pressing artificial
epochs.add_proj(proj)
print(epochs)

epochs.save(OUTPUT_DIR / 'epochs-clean-ERP-1-epo.fif', overwrite=True)

# Check if epochs is all right
evoked = epochs.average()
fig = evoked.plot_joint(show=False, title='evoked-clean-ERP')
fig.savefig(OUTPUT_DIR / 'evoked-clean-ERP-1.png')
plt.close(fig)
# plt.show()


# %%
evoked = evoked1.copy()
evoked.data = evoked.data - evoked2.data
# evoked1.plot_joint(show=False, title='evoked-1')
# evoked2.plot_joint(show=False, title='evoked-2')
fig = evoked.plot_joint(show=False, title='evoked')
fig.savefig(OUTPUT_DIR / 'evoked-1-remove-keypress-artificial.png')
plt.close(fig)

evoked.add_proj(proj)
fig = evoked.plot_joint(show=False, title='evoked(proj)')
fig.savefig(OUTPUT_DIR / 'evoked-1-remove-keypress-artificial-proj.png')
plt.close(fig)
# plt.show()

# %% ---- 2026-02-02 ------------------------
# Pending

# %% ---- 2026-02-02 ------------------------
# Pending

# %%
