"""
File: 2.average.subjects.py
Author: Chuncheng Zhang
Date: 2026-02-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Average all subjects.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-25 ------------------------
# Requirements and constants
from itertools import product
from util.easy_imports import *

# %%
MODE = 'EEG'

if len(sys.argv) > 1:
    _, MODE = sys.argv

logger.info(f'Run {__file__} for {MODE=}')

# %%
DATA_DIR = Path(f'output/step-1')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist.'

OUTPUT_DIR = Path(f'output/step-1-subjects-average/{MODE}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-02-25 ------------------------
# Function and class


# %% ---- 2026-02-25 ------------------------
# Play ground
for evt, epo in product(['1', '2', '3'], ['epo', 'hilbert-epo']):
    fif_files = sorted(list(DATA_DIR.rglob(f'{MODE}*/epochs-{evt}-{epo}.fif')))
    print(fif_files)

    epochs_array = [mne.read_epochs(f) for f in fif_files]

    epochs = mne.concatenate_epochs(epochs_array, on_mismatch='ignore')
    print(epochs)

    evoked = epochs.average()
    evoked.save(OUTPUT_DIR / f'{evt}-{epo}-ave.fif', overwrite=True)

    fig = evoked.plot_joint(show=False)
    fig.savefig(OUTPUT_DIR / f'{evt}-{epo}.png')
    plt.close(fig)

# %% ---- 2026-02-25 ------------------------
# Pending
logger.info(f'Finished {__file__} for {MODE=}')

exit(0)

# %% ---- 2026-02-25 ------------------------
# Pending
