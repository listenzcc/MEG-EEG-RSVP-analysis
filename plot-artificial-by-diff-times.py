"""
File: plot-artificial-by-diff-times.py
Author: Chuncheng Zhang
Date: 2026-06-01
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Plot artificial by different diff times.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-01 ------------------------
# Requirements and constants
from util.easy_imports import *

# %%
# MODE = 'EEG'
MODE = 'MEG'


# %%
DATA_DIR = Path('./output/step-1')


# %% ---- 2026-06-01 ------------------------
# Function and class


# %% ---- 2026-06-01 ------------------------
# Play ground
table = pd.read_csv('output/diff-times.csv')
table

# %%
subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(subjects)

# %%
threshold = 0.4  # seconds

subject = subjects[0]

epochs_quick_array = []
epochs_slow_array = []

dev_head_t = None

for subject in tqdm(subjects):
    mode, subj = subject.split('-')
    print(subject, mode, subj)

    if not mode == MODE:
        continue

    epochs3 = mne.read_epochs(
        DATA_DIR / subject / 'epochs-3-epo.fif', preload=True)
    print(epochs3)

    # Always use the same dev_head_t for all epochs to ensure they are in the same coordinate system
    if dev_head_t is None:
        dev_head_t = epochs3.info['dev_head_t']
    else:
        epochs3.info['dev_head_t'] = dev_head_t

    df = table[(table['subject'] == subj) & (table['mode'] == mode)]
    print(df.head())

    df_quick = df[df['delay'] < threshold]
    df_slow = df[df['delay'] >= threshold]
    epochs_quick = epochs3[df_quick['index'].values]
    epochs_slow = epochs3[df_slow['index'].values]
    print(epochs_quick)
    print(epochs_slow)

    epochs_quick_array.append(epochs_quick)
    epochs_slow_array.append(epochs_slow)

epochs_quick = mne.concatenate_epochs(epochs_quick_array)
epochs_slow = mne.concatenate_epochs(epochs_slow_array)

print(epochs_quick)
print(epochs_slow)

# %%
evoked_quick = epochs_quick.average()
evoked_slow = epochs_slow.average()

fig = evoked_quick.plot_joint(
    title='Quick Keypress (Delay < 0.4s)', show=False)
fig.savefig(f'output/{MODE}-evoked-quick-by-diff-times.png')
# plt.show()
fig = evoked_slow.plot_joint(
    title='Slow Keypress (Delay >= 0.4s)', show=False)
fig.savefig(f'output/{MODE}-evoked-slow-by-diff-times.png')
# plt.show()

# %% ---- 2026-06-01 ------------------------
# Pending


# %% ---- 2026-06-01 ------------------------
# Pending
