
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
SUBJ = 'S03'
MODE = 'MEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
OUTPUT_DIR = Path(f'output/step-1/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %%


def read_raw(path: Path):
    raw = mne.io.read_raw_fif(path, preload=False)
    if MODE == 'MEG':
        raw.pick('mag')
    elif MODE == 'EEG':
        raw.pick([e for e in raw.ch_names if e not in ['CB1', 'CB2']])
        montage = mne.channels.read_dig_fif('output/eeg-montage-dig.fif')
        montage.ch_names = [e.upper() for e in montage.ch_names]
        raw.rename_channels({e: e.upper() for e in raw.ch_names})
        raw.set_montage(montage)
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
logger.info(raw.info)

# Mark the '4' events for dirty '2' events


def remark_events(events, event_id):
    event_id = {str(k): k for k in [1, 2, 3, 4]}
    events_target = np.array([e for e in events if e[-1] in [1, 3]])
    events_normal = np.array([e for e in events if e[-1] in [2]])
    for rec in tqdm(events_target):
        t, _, e = rec
        m = (events_normal[:, 0] < t+1500) * (events_normal[:, 0] > t-500)
        events_normal[m, -1] = 4

    events = np.concat([events_target, events_normal])
    events = np.array(sorted(events, key=lambda e: e[0]))
    return events, event_id


events, event_id = remark_events(events, event_id)
detail = {e: events[events[:, -1] == e].shape for e in event_id.values()}
logger.debug(f'Remark events to 1, 2, 3, 4, {detail=}')


# %%
# Plot events
# mne.viz.plot_events(events, raw.info['sfreq'])
# plt.show()

# %%
tmin, tmax = -0.5, 1.5
l_freq, h_freq = 0.1, 40

if MODE == 'MEG':
    reject = dict(
        mag=4e-12,      # unit: T (magnetometers)
    )
elif MODE == 'EEG':
    reject = dict(
        eeg=40e-6,      # unit: V (EEG channels)
    )
else:
    raise ValueError(f'Incorrect {MODE=}')

epochs = mne.Epochs(raw, events, event_id,
                    tmin=tmin, tmax=tmax,
                    decim=int(raw.info['sfreq'] / 200),
                    reject=reject,
                    )

# Only interested in 1, 2, 3 events
epochs = mne.concatenate_epochs([epochs['1'], epochs['2'], epochs['3']])
logger.debug(f'Generated {epochs=}')

# It may take long since the epochs are so many
epochs.load_data()
epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

# Save 1, 3 epochs
fname = OUTPUT_DIR / f'epochs-1-3-epo.fif'
_epochs = mne.concatenate_epochs([epochs['1'], epochs['3']])
_epochs.save(fname)
logger.debug(f'Saved {fname=}, {_epochs=}')

# Plot and save evoked
for evt in ['1', '2', '3']:
    evoked = epochs[evt].average()
    evoked.save(OUTPUT_DIR / f'evoked-{evt}-ave.fif')

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(OUTPUT_DIR / f'evoked-{evt}.png')
    plt.close(fig)

# %%
hilbert = epochs.copy().apply_hilbert(envelope=True)

# Save 1, 3 hilbert epochs
fname = OUTPUT_DIR / f'epochs-1-3-hilbert-epo.fif'
_hilbert = mne.concatenate_epochs([hilbert['1'], hilbert['3']])
_hilbert.save(fname)
logger.debug(f'Saved {fname=}, {_hilbert=}')

# Plot and save hilbert evoked
for evt in ['1', '2', '3']:
    evoked = hilbert[evt].average()
    evoked.save(OUTPUT_DIR / f'hilbert-evoked-{evt}-ave.fif')

    evoked.apply_baseline()
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
