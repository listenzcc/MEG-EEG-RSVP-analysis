"""
File: 1.raw.to.epochs.py
Author: Chuncheng Zhang
Date: 2026-02-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Read raw and convert into epochs.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-25 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
RAW_DIR = Path('data/RSVP_dataset/processed_data') / f'{MODE}_{SUBJ}'
assert RAW_DIR.exists(), f'{RAW_DIR} does not exist'

OUTPUT_DIR = Path(f'output/step-1/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-02-25 ------------------------
# Function and class


def read_raw(path: Path):
    '''
    Read raw data from the given path and pick relevant channels based on the MODE (MEG or EEG).
    The raw data is rename_channels to ensure consistency, and for EEG data, the montage is set.

    Args:
        path: Path to the raw data file

    Returns:
        raw: MNE Raw object with the relevant channels picked and montage set (for EEG)
    '''
    raw = mne.io.read_raw_fif(path, preload=False)
    if MODE == 'MEG':
        # Pick magnetometers for MEG data
        # Also rename channels to remove the '-4503' suffix for consistency
        raw.pick('mag')
        raw.rename_channels({e: e.replace('-4503', '') for e in raw.ch_names})
    elif MODE == 'EEG':
        # Pick EEG channels, excluding CB1 and CB2
        # Also set montage for EEG data
        # Note: The montage file should be prepared in advance and should match the channel names in the raw data
        # Convert the channel names to uppercase to ensure consistency with the montage
        raw.pick([e for e in raw.ch_names if e not in ['CB1', 'CB2']])
        montage = mne.channels.read_dig_fif('asset/eeg-montage-dig.fif')
        montage.ch_names = [e.upper() for e in montage.ch_names]
        raw.rename_channels({e: e.upper() for e in raw.ch_names})
        raw.set_montage(montage)
    return raw


def remark_events(events, event_id):
    '''
    Remark events according to the temporal proximity of target and normal events.
    Target events are those with event_id 1 and 3,
    while normal events are those with event_id 2.
    If a normal event occurs within 500 ms before or 1500 ms after a target event,
    it is re-labeled as 4.

    Args:
        events: numpy array of shape (n_events, 3), where each row is [time, 0, event_id]
        event_id: dictionary mapping event labels to their corresponding integer codes

    Returns:
        events: numpy array of shape (n_events, 3) with updated event_id for normal events that are close to target events
        event_id: updated dictionary with new event_id mapping
    '''
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


# %% ---- 2026-02-25 ------------------------
# Play ground
# Find files after ICA
files = list(RAW_DIR.glob('block_*_ica-raw.fif'))
files.sort()
print(files)

# Skip the first file which is incorrect
raws = [read_raw(e) for e in files[1:]]

# 'ignore' to ignore when the device-to-head transformation differs between instances.
raw = mne.concatenate_raws(raws, on_mismatch='ignore')
print(raw.info)

# Read events and event_id from annotations
# Also remark events according to the temporal proximity of target and normal events
events, event_id = mne.events_from_annotations(raw)
events, event_id = remark_events(events, event_id)

fig = mne.viz.plot_events(
    events, raw.info['sfreq'], event_id=event_id, show=False)
fig.savefig(OUTPUT_DIR / 'events.png')
plt.close(fig)

# %% ---- 2026-02-25 ------------------------
# Generate epochs
tmin, tmax = -0.5, 1.5
l_freq, h_freq = 0.1, 40

if MODE == 'MEG':
    reject = dict(
        mag=4e-12,      # unit: T (magnetometers)
    )
elif MODE == 'EEG':
    reject = dict(
        # eeg=40e-6,      # unit: V (EEG channels)
    )
else:
    raise ValueError(f'Incorrect {MODE=}')

# Ready to generate epochs
# Also downsample by decimating the data to 200 Hz to reduce memory usage and speed up processing
epochs = mne.Epochs(
    raw, events, event_id,
    tmin=tmin, tmax=tmax,
    decim=int(raw.info['sfreq'] / 200),
    reject=reject,
)

# Only interested in 1, 2, 3 events
epochs = mne.concatenate_epochs([epochs['1'], epochs['2'], epochs['3']])
logger.debug(f'Generated {epochs=}')

# It may take long since the epochs are so many
epochs.load_data()

# For EEG data, set the average reference
if MODE == 'EEG':
    epochs = epochs.set_eeg_reference(ref_channels='average')

epochs.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=n_jobs)

# %% ---- 2026-02-25 ------------------------
# Save epochs
# Save epochs for 1, 2, 3 events separately
# Plot and save evoked
for evt in ['1', '2', '3']:
    fname = OUTPUT_DIR / f'epochs-{evt}-epo.fif'
    epochs[evt].save(fname, overwrite=True)
    logger.debug(f'Saved {fname=}, {epochs[evt]=}')

    fname = OUTPUT_DIR / f'epochs-{evt}-ave.fif'
    evoked = epochs[evt].average()
    evoked.save(fname, overwrite=True)
    logger.debug(f'Saved {fname=}')

    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname.with_suffix('.png'))
    plt.close(fig)

# %%
# Apply Hilbert transform to get the envelope of the signal in the time domain, which can be useful for analyzing the amplitude modulation of the signal.
hilbert = epochs.copy().apply_hilbert(envelope=True)

# Save hilbert epochs for 1, 2, 3 events separately
for evt in ['1', '2', '3']:
    fname = OUTPUT_DIR / f'epochs-{evt}-hilbert-epo.fif'
    hilbert[evt].save(fname, overwrite=True)
    logger.debug(f'Saved {fname=}')

    fname = OUTPUT_DIR / f'epochs-{evt}-hilbert-ave.fif'
    evoked = hilbert[evt].average()
    evoked.save(fname, overwrite=True)
    logger.debug(f'Saved {fname=}')

    evoked.apply_baseline()
    fig = evoked.plot_joint(title=f'{evt=}', show=False)
    fig.savefig(fname.with_suffix('.png'))
    plt.close(fig)

logger.info(f'Finished {__file__} for {SUBJ=}, {MODE=}')
exit(0)
# %%
