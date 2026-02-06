"""
File: 4.erp.of.roi.channels.py
Author: Chuncheng Zhang
Date: 2026-02-06
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Analysis ERPs of interested channels.

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
MODE = 'MEG'

# %%
DATA_DIR = Path('./output/step-2-with-filter')

OUTPUT_DIR = Path('./output/step-4')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-02-06 ------------------------
# Function and class


# %% ---- 2026-02-06 ------------------------
# Play ground
fif_files = list(DATA_DIR.rglob(f'{MODE}-S*/epochs-clean-ERP-1-epo.fif'))
print(fif_files)
epochs_array = [mne.read_epochs(e) for e in fif_files]
print(epochs_array)

# Concat to generate the epochsArray
info = epochs_array[0].info
event_id = epochs_array[0].event_id
tmin = epochs_array[0].tmin
events_array = [e.events for e in epochs_array]
m = np.max([np.max(e) for e in events_array])
for i, e in enumerate(events_array):
    e[:, 0] += int(i * m*2)

data_array = [e.get_data() for e in epochs_array]
events = np.concat(events_array)
data = np.concat(data_array)
print(f'{events.shape=}, {data.shape=}')
epochs = mne.EpochsArray(data, info=info, events=events,
                         event_id=event_id, tmin=tmin)
map_ch_names = {e: e.split('-')[0] for e in epochs.ch_names}
epochs.rename_channels(map_ch_names)
print(epochs)

# %%
# epochs = epochs.copy().apply_function(
#     fun=lambda x: mne.filter.detrend(x, order=1),
#     n_jobs=n_jobs
# )
# print(epochs)

# %%
# evoked = epochs.average()
# evoked.detrend(order=1)
# evoked.apply_baseline()
# fig = evoked.plot_joint(show=False)
# plt.show()

# %%
evoked = epochs.average()
evoked.detrend(order=1)
evoked.apply_baseline()

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
evoked.plot_sensors(axes=ax, show_names=True, show=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{MODE}-sensors.png')
plt.close(fig)

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
evoked.plot_topo(axes=ax, show=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{MODE}-topo.png')
plt.close(fig)

if MODE == 'EEG':
    evoked = evoked.pick(
        [e for e in evoked.ch_names
         if any([e.startswith(x) for x in ['O']])])

if MODE == 'MEG':
    evoked = evoked.pick(['MLO42', 'MLO43', 'MLO51', 'MLO52', 'MLO53'])

fig = evoked.plot_joint(show=False)
fig.savefig(OUTPUT_DIR / f'{MODE}-joinplot.png')
plt.close(fig)


# %%

# %% ---- 2026-02-06 ------------------------
# Pending


# %% ---- 2026-02-06 ------------------------
# Pending
