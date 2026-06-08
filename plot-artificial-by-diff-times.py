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
MODE = 'EEG'
MODE = 'MEG'


# %%
DATA_DIR = Path('./output/step-1')
OUTPUT_DIR = Path('./output/artificial-by-diff-times')
OUTPUT_DIR.mkdir(exist_ok=True)


# %% ---- 2026-06-01 ------------------------
# Function and class


# %% ---- 2026-06-01 ------------------------
# Play ground
table_target = pd.read_csv('output/diff-times-target.csv')
table_keypress = pd.read_csv('output/diff-times-keypress.csv')

# %%
subjects = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
print(subjects)

# %%
threshold = 0.4  # seconds

subject = subjects[0]

epochs_quick_target_array = []
epochs_slow_target_array = []
epochs_quick_keypress_array = []
epochs_slow_keypress_array = []

dev_head_t_1 = None
dev_head_t_3 = None

for subject in tqdm(subjects):
    mode, subj = subject.split('-')
    print(subject, mode, subj)

    if not mode == MODE:
        continue

    # Target
    epochs1 = mne.read_epochs(
        DATA_DIR / subject / 'epochs-1-epo.fif', preload=True)
    # Keypress
    epochs3 = mne.read_epochs(
        DATA_DIR / subject / 'epochs-3-epo.fif', preload=True)
    print(epochs1, epochs3)

    # Always use the same dev_head_t for all epochs to ensure they are in the same coordinate system
    if dev_head_t_1 is None:
        dev_head_t_1 = epochs1.info['dev_head_t']
    else:
        epochs1.info['dev_head_t'] = dev_head_t_1

    # Always use the same dev_head_t for all epochs to ensure they are in the same coordinate system
    if dev_head_t_3 is None:
        dev_head_t_3 = epochs3.info['dev_head_t']
    else:
        epochs3.info['dev_head_t'] = dev_head_t_3

    df_target = table_target[(table_target['subject'] == subj) & (
        table_target['mode'] == mode)]
    df_keypress = table_keypress[(table_keypress['subject'] == subj) & (
        table_keypress['mode'] == mode)]
    print(df_target.head())
    print(df_keypress.head())

    # Target
    df_quick = df_target[df_target['delay'] < threshold]
    df_slow = df_target[df_target['delay'] >= threshold]
    epochs_quick = epochs1[df_quick['index'].values]
    epochs_slow = epochs1[df_slow['index'].values]
    print(epochs_quick)
    print(epochs_slow)
    epochs_quick_target_array.append(epochs_quick)
    epochs_slow_target_array.append(epochs_slow)

    # Keypress
    df_quick = df_keypress[df_keypress['delay'] < threshold]
    df_slow = df_keypress[df_keypress['delay'] >= threshold]
    epochs_quick = epochs3[df_quick['index'].values]
    epochs_slow = epochs3[df_slow['index'].values]
    print(epochs_quick)
    print(epochs_slow)
    epochs_quick_keypress_array.append(epochs_quick)
    epochs_slow_keypress_array.append(epochs_slow)

epochs_target_quick = mne.concatenate_epochs(epochs_quick_target_array)
epochs_target_slow = mne.concatenate_epochs(epochs_slow_target_array)
print(epochs_target_quick)
print(epochs_target_slow)

epochs_keypress_quick = mne.concatenate_epochs(epochs_quick_keypress_array)
epochs_keypress_slow = mne.concatenate_epochs(epochs_slow_keypress_array)
print(epochs_keypress_quick)
print(epochs_keypress_slow)

# %%
evoked_proj_quick = epochs_target_quick.average()
evoked_proj_slow = epochs_target_slow.average()

data_dir = Path(f'output/step-1-subjects-average/{MODE}')
evoked2 = mne.read_evokeds(data_dir / '2-epo-ave.fif')[0]
evoked3 = mne.read_evokeds(data_dir / '3-epo-ave.fif')[0]

if MODE == 'EEG':
    kwargs = dict(n_grad=0, n_mag=0, n_eeg=1)
elif MODE == 'MEG':
    kwargs = dict(n_grad=0, n_mag=3, n_eeg=0)

proj2 = mne.compute_proj_evoked(evoked2, **kwargs)
proj3 = mne.compute_proj_evoked(evoked3, **kwargs)
proj = proj2 + proj3

evoked_proj_quick.add_proj(proj)
evoked_proj_quick.apply_proj()
evoked_proj_slow.add_proj(proj)
evoked_proj_slow.apply_proj()

for evoked_quick, evoked_slow, cond in zip(
    (epochs_target_quick.average(),
     epochs_keypress_quick.average(),
     evoked_proj_quick,),
    (epochs_target_slow.average(),
     epochs_keypress_slow.average(),
     evoked_proj_slow),
    ('target',
     'keypress',
     'proj')
):
    evoked_quick.save(
        OUTPUT_DIR / f'{MODE}-evoked-{cond}-quick-by-diff-times-ave.fif', overwrite=True)
    evoked_slow.save(
        OUTPUT_DIR / f'{MODE}-evoked-{cond}-slow-by-diff-times-ave.fif', overwrite=True)

    fig = evoked_quick.plot_joint(
        title='Quick Keypress (Delay < 0.4s)', show=False)
    fig.savefig(OUTPUT_DIR / f'{MODE}-evoked-{cond}-quick-by-diff-times.png')
    # plt.show()

    fig = evoked_slow.plot_joint(
        title='Slow Keypress (Delay >= 0.4s)', show=False)
    fig.savefig(OUTPUT_DIR / f'{MODE}-evoked-{cond}-slow-by-diff-times.png')
    # plt.show()

# %% ---- 2026-06-01 ------------------------
# Pending

# %% ---- 2026-06-01 ------------------------
# Pending
