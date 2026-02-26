"""
File: 1.2.remove.artificial.py
Author: Chuncheng Zhang
Date: 2026-02-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read epochs and remove artificial.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-26 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
MODE = 'EEG'

if len(sys.argv) > 1:
    _, MODE = sys.argv

logger.info(f'Run {__file__} for {MODE=}')

# %%
DATA_DIR = Path(f'output/step-1-subjects-average/{MODE}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist.'

OUTPUT_DIR = Path(f'output/step-1-subjects-average-proj/{MODE}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-02-26 ------------------------
# Function and class


# %% ---- 2026-02-26 ------------------------
# Play ground
evoked1 = mne.read_evokeds(DATA_DIR / '1-epo-ave.fif')[0]
evoked2 = mne.read_evokeds(DATA_DIR / '2-epo-ave.fif')[0]
evoked3 = mne.read_evokeds(DATA_DIR / '3-epo-ave.fif')[0]

print(evoked1, evoked2, evoked3)

# %%

if MODE == 'EEG':
    kwargs = dict(n_grad=0, n_mag=0, n_eeg=1)
elif MODE == 'MEG':
    kwargs = dict(n_grad=0, n_mag=3, n_eeg=0)

proj2 = mne.compute_proj_evoked(evoked2, **kwargs)
proj3 = mne.compute_proj_evoked(evoked3, **kwargs)
proj = proj2 + proj3

evoked = evoked1.copy()
evoked.add_proj(proj)
evoked.apply_proj()

evoked.save(OUTPUT_DIR / f'1-withproj-epo-ave.fif', overwrite=True)
mne.write_proj(OUTPUT_DIR / f'2-3-proj.fif', proj, overwrite=True)

# fig = evoked1.plot_joint(show=False)
fig = evoked.plot_joint(show=False)
fig.savefig(OUTPUT_DIR / f'1-withproj-epo-ave.png')
plt.close(fig)
# plt.show()

logger.info(f'Finished {__file__} for {MODE=}')

# %% ---- 2026-02-26 ------------------------
# Pending


# %% ---- 2026-02-26 ------------------------
# Pending
