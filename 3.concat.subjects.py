"""
File: 3.concat.subjects.py
Author: Chuncheng Zhang
Date: 2026-02-05
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Average the subjects.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-02-05 ------------------------
# Requirements and constants
from util.easy_imports import *

import tensorly as tl
from tensorly.decomposition import tucker

# %%
MODE = 'MEG'

if len(sys.argv) > 1:
    _, MODE = sys.argv

logger.info(f'Run {__file__} for {MODE=}')

# %%
DATA_DIR = Path('./output/step-2')

OUTPUT_DIR = Path('./output/step-3')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-02-05 ------------------------
# Function and class


# %% ---- 2026-02-05 ------------------------
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
print(epochs)

# %% ---- 2026-02-05 ------------------------
# Pending
evoked = epochs.average()
evoked.detrend(order=1)
evoked.apply_baseline()
fig = evoked.plot_joint(show=False)
fig.savefig(OUTPUT_DIR / f'evoked-jointplot-{MODE}.png')
plt.close(fig)
# plt.show()

# %%
epochs = epochs.copy().apply_function(
    fun=lambda x: mne.filter.detrend(x, order=1),
    n_jobs=n_jobs
)
epochs

# %% ---- 2026-02-05 ------------------------
# Pending
# tl.set_backend('numpy')
tl.set_backend('pytorch')

X = epochs.get_data(copy=False)

N, C, T = X.shape
r_c = 10   # 空间秩
r_t = 15   # 时间秩

r_c = 8
r_t = 6

# ranks: (trial_rank, channel_rank, time_rank)
# trial_rank = None → 不在该维度做压缩
# ! It is slow
tX = tl.tensor(X)
tX.to('cuda')
core, factors = tucker(
    tX,
    rank=[None, r_c, r_t],
    init='svd',
    tol=1e-6,
    n_iter_max=200
)

# factors:
# factors[0] = None
# factors[1] = U_c ∈ R^{C × r_c}
# factors[2] = U_t ∈ R^{T × r_t}

U_c = np.array(factors[1])
U_t = np.array(factors[2])
G = np.array(core)            # shape: (N, r_c, r_t)

# %%
# Save
with open(OUTPUT_DIR / f'decomp-{MODE}.npy', 'wb') as f:
    np.save(f, U_c)
    np.save(f, U_t)
    np.save(f, G)

# Read
# with open(OUTPUT_DIR / f'decomp-{MODE}.npy', 'rb') as f:
#     U_c = np.load(f)
#     U_t = np.load(f)
#     G = np.load(f)

print(f'{U_c.shape=}, {U_t.shape=}, {G.shape=}')

# %%
times = evoked.times
cols = 3
rows = int(np.ceil(r_t / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 8/cols*rows))
for ts, ax in zip(U_t.T, axes.ravel()):
    ax.plot(times, ts)
fig.savefig(OUTPUT_DIR / f'ts-{MODE}.png')
plt.close(fig)
# plt.show()

# %%
fig, axes = plt.subplots(
    1, r_c+1, figsize=(12, 3),
    gridspec_kw={
        'width_ratios': [1] * r_c + [0.2]  # 最后一列给 legend
    })

i = 0  # 第 i 个空间模式
for i in range(r_c):
    ax = axes[i]

    topo = U_c[:, i]    # shape: (62,)

    topo = topo / np.max(np.abs(topo))

    evoked = mne.EvokedArray(
        topo[:, np.newaxis],  # shape: (n_channels, n_times=1)
        info,
        tmin=0
    )

    evoked.plot_topomap(
        times=0,
        scalings=1,
        cmap='RdBu_r',
        contours=6,
        time_format='Map %d' % i,
        axes=[ax, axes[r_c]],
        show=False
    )

fig.tight_layout()
fig.savefig(OUTPUT_DIR / f'topo-{MODE}.png')
plt.close(fig)
# plt.show()

# %%
dir(tl)


# %%
