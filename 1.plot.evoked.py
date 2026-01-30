
"""
File: 1.plot.evoked.py
Author: Chuncheng Zhang
Date: 2026-01-26
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read raw files, concat them and plot evoked.

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

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
OUTPUT_DIR = Path(f'output/example/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

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

# Beware that the 1st file is incorrect.
raws = [read_raw(e) for e in block_files[1:]]

if MODE == 'MEG':
    dev_head_t = raws[0].info['dev_head_t']
    for raw in raws:
        raw.info['dev_head_t'] = dev_head_t
print(raws)


# %%
raw = mne.concatenate_raws(raws)

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
epochs = mne.concatenate_epochs([epochs['1'], epochs['3']])
print(epochs)
epochs.load_data()
epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)
epochs.save(OUTPUT_DIR / f'1-3-epo.fif')
print(epochs)

# %%
print(epochs)
for evt in ['1', '3']:
    evoked = epochs[evt].average()
    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(OUTPUT_DIR / f'evoked-{evt}.png')
    plt.close(fig)

# %%
hilbert = epochs.copy().apply_hilbert(envelope=True)
hilbert.save(OUTPUT_DIR / f'1-3-hilbert-epo.fif')
hilbert.apply_baseline()
print(hilbert)

for evt in ['1', '3']:
    evoked = hilbert[evt].average()
    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(OUTPUT_DIR / f'hilbert-evoked-{evt}.png')
    plt.close(fig)

# %% ---- 2026-01-26 ------------------------
# Function and class


# %% ---- 2026-01-26 ------------------------
# Play ground


# %% ---- 2026-01-26 ------------------------
# Pending


# %% ---- 2026-01-26 ------------------------
# Pending
