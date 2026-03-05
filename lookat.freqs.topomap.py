"""
File: lookat.freqs.topomap.py
Author: Chuncheng Zhang
Date: 2026-03-04
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Have a look at the topomap of freqs of interests.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-04 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
DATA_DIR = Path(f'output/step-1-subjects-average')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist.'

DATA_DIR1 = Path(f'output/step-1-subjects-average-proj')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist.'

# %%
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-03-04 ------------------------
# Function and class


# %% ---- 2026-03-04 ------------------------
# Play ground

# %%
# Plot 2 event
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

evt = '2'
for i, mode in enumerate(['EEG', 'MEG']):
    evoked = mne.read_evokeds(
        DATA_DIR / mode / f'{evt}-epo-ave.fif')[0]

    freqs = np.arange(2, 31, 1)
    t = evoked.times[-1] - evoked.times[0]
    n_cycles = [int(f*t*0.5) for f in freqs]

    tfr = mne.time_frequency.tfr_morlet(
        evoked, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, return_itc=False)

    # I want the values of 10Hz
    ax = axes[0, i]
    values = np.mean(
        np.mean(tfr.data[:, [e < 12 and e > 8 for e in freqs], :], axis=1), axis=1)

    vlim = [np.min(values), np.max(values)]

    im, cn = mne.viz.plot_topomap(
        values, evoked.info, vlim=vlim, axes=ax, show=False)
    ax.set_title(f'{mode} ({evt=}) (10Hz)')
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)

    # I want the values of 20Hz
    ax = axes[1, i]
    values = np.mean(
        np.mean(tfr.data[:, [e < 22 and e > 18 for e in freqs], :], axis=1), axis=1)

    vlim = [np.min(values), np.max(values)]

    im, cn = mne.viz.plot_topomap(
        values, evoked.info, vlim=vlim, axes=ax, show=False)
    ax.set_title(f'{mode} ({evt=}) (20Hz)')
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{evt}-freqs-topomap.png', dpi=300)
plt.show()

# %%
# Plot 3 event
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

evt = '3'
for i, mode in enumerate(['EEG', 'MEG']):
    evoked = mne.read_evokeds(
        DATA_DIR / mode / f'{evt}-epo-ave.fif')[0]

    freqs = np.arange(2, 31, 1)
    t = evoked.times[-1] - evoked.times[0]
    n_cycles = [int(f*t*0.5) for f in freqs]

    tfr = mne.time_frequency.tfr_morlet(
        evoked, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, return_itc=False)

    # I want the values of <10 Hz
    ax = axes[i]
    values = np.mean(
        np.mean(tfr.data[:, [e < 10 for e in freqs], :], axis=1), axis=1)

    vlim = [np.min(values), np.max(values)]

    im, cn = mne.viz.plot_topomap(
        values, evoked.info, vlim=vlim, axes=ax, show=False)
    ax.set_title(f'{mode} ({evt=}) (<10Hz)')
    fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{evt}-freqs-topomap.png', dpi=300)
plt.show()

# %%
# Plot 1 event
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

evt = '1'

for i, mode in enumerate(['EEG', 'MEG']):
    for j in [0, 1]:
        if j == 0:
            evoked = mne.read_evokeds(
                DATA_DIR / mode / f'{evt}-epo-ave.fif')[0]
        elif j == 1:
            evoked = mne.read_evokeds(
                DATA_DIR1 / mode / f'{evt}-withproj-epo-ave.fif')[0]

        freqs = np.arange(2, 31, 1)
        t = evoked.times[-1] - evoked.times[0]
        n_cycles = [int(f*t*0.5) for f in freqs]

        tfr = mne.time_frequency.tfr_morlet(
            evoked, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, return_itc=False)

        # I want the values of <10 Hz
        ax = axes[j, i]
        values = np.mean(
            np.mean(tfr.data[:, [e < 10 for e in freqs], :], axis=1), axis=1)

        vlim = [np.min(values), np.max(values)]

        im, cn = mne.viz.plot_topomap(
            values, evoked.info, vlim=vlim, axes=ax, show=False)
        ax.set_title(f'{mode} ({evt=}) (<10Hz)')
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{evt}-freqs-topomap.png', dpi=300)
plt.show()

# %%
# Plot detail of 1 (projected) event
times = np.arange(-0.1, 1.1, 0.1)
n_times = len(times)
fig, axes = plt.subplots(2, n_times, figsize=(n_times*2, 4))

for i, mode in enumerate(['MEG', 'EEG']):
    evoked = mne.read_evokeds(
        DATA_DIR1 / mode / f'{evt}-withproj-epo-ave.fif')[0]

    freqs = np.arange(2, 31, 1)
    t = evoked.times[-1] - evoked.times[0]
    n_cycles = [int(f*t*0.5) for f in freqs]
    tfr = mne.time_frequency.tfr_morlet(
        evoked, freqs=freqs, n_cycles=n_cycles, n_jobs=n_jobs, return_itc=False)
    values = np.mean(
        np.mean(tfr.data[:, [e < 10 for e in freqs], :], axis=1), axis=1)
    vlim = [np.min(values)*0.8, np.max(values)*3]

    t_segs = np.linspace(evoked.times[0], evoked.times[-1], n_times+1)
    for j, t1 in enumerate(times):
        t2 = t1+0.1
        ax = axes[i, j]
        _tfr = tfr.copy()
        _tfr.crop(tmin=t1, tmax=t2)
        values = np.mean(
            np.mean(_tfr.data[:, [e < 10 for e in freqs], :], axis=1), axis=1)
        im, cn = mne.viz.plot_topomap(
            values, evoked.info, vlim=vlim, axes=ax, show=False)
        ax.set_title(f'{t1:.2f} ~ {t2:.2f}')
        fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.5)

suptitle = 'Time evolution of 1(proj) of MEG and EEG'
fig.suptitle(suptitle)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'{suptitle}.png')
plt.show()


# %% ---- 2026-03-04 ------------------------
# Pending


# %% ---- 2026-03-04 ------------------------
# Pending

# %%
