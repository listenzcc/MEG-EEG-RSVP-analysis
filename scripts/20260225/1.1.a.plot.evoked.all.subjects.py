"""
File: 1.1.a.plot.evoked.all.subjects.py
Author: Chuncheng Zhang
Date: 2026-02-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Average all the subjects.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-06 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
MODE = 'EEG'

if len(sys.argv) > 1:
    _, MODE = sys.argv


logger.info(f'Run {__file__} for {MODE=}')

# %%
DATA_DIR = Path(f'output/step-1')

OUTPUT_DIR = Path(f'output/step-1-average-all-subjects')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% ---- 2026-02-06 ------------------------
# Function and class


# %% ---- 2026-02-06 ------------------------
# Play ground
fif_files = sorted(list(DATA_DIR.rglob(f'{MODE}*/epochs-1-3-epo.fif')))
fif_files.extend(sorted(list(DATA_DIR.rglob(f'{MODE}*/epochs-2-epo.fif'))))
print(fif_files)

epochs_array = [mne.read_epochs(f) for f in fif_files]

epochs = mne.concatenate_epochs(epochs_array, on_mismatch='ignore')
print(epochs)

# %% ---- 2026-02-06 ------------------------
# Pending
for evt in ['1', '2', '3']:
    evoked = epochs[evt].average()
    evoked.save(OUTPUT_DIR / f'{MODE}-{evt}-ave.fif', overwrite=True)

    fig = evoked.plot_joint(show=False)
    fig.savefig(OUTPUT_DIR / f'{MODE}-{evt}.png')
    plt.title(f'{MODE}-{evt}')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    evoked.plot_topo(axes=ax, show=False)
    ax.set_title(f'{MODE}-{evt}')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f'{MODE}-{evt}-topo.png')
    plt.close(fig)


# %% ---- 2026-02-06 ------------------------
# Pending
