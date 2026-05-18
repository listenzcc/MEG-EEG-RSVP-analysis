"""
File: compare-with-without-artificial-removal.py
Author: Chuncheng Zhang
Date: 2026-05-15
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Compare ERPs with & without artificial removal.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-05-15 ------------------------
# Requirements and constants
from util.easy_imports import *
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.datasets import fetch_fsaverage

# %% ---- 2026-05-15 ------------------------
# Function and class


def source_estimation_evoked(evoked, method='MNE', snr=3.0, loose=0.2, depth=0.8, baseline=(None, 0.0)):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""
    evoked = evoked.copy()

    # Ensure fsaverage is available and use the standard fsaverage BEM/source space.
    fs_dir = fetch_fsaverage(verbose=False)
    subjects_dir = os.path.dirname(fs_dir)
    trans = 'fsaverage'
    src_fname = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem_fname = os.path.join(
        fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

    # If EEG data has no montage, assign a standard montage.
    if evoked.get_montage() is None:
        montage = mne.channels.make_standard_montage('standard_1020')
        evoked.set_montage(montage)

    # Determine channel types for the forward model.
    # print(mne.pick_types(evoked.info))
    # print(evoked.info)
    # use_eeg = bool(mne.pick_types(evoked.info, eeg=True, meg=False))
    # use_meg = bool(mne.pick_types(evoked.info, meg=True, eeg=False))
    use_eeg = len(mne.pick_types(evoked.info, eeg=True, meg=False)) > 0
    use_meg = len(mne.pick_types(evoked.info, meg=True, eeg=False)) > 0
    print(f'{use_eeg=}, {use_meg=}')

    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(
        evoked.info,
        trans=trans,
        src=src,
        bem=bem_fname,
        eeg=use_eeg,
        meg=use_meg,
    )

    # Use baseline data to estimate noise covariance.
    tmin, tmax = baseline
    if tmin is None:
        tmin = evoked.tmin
    if tmax is None:
        tmax = min(0.0, evoked.times[-1])

    baseline_evoked = evoked.copy().crop(tmin, tmax)
    baseline_data = baseline_evoked.data
    cov = mne.Covariance(
        data=np.cov(baseline_data),
        names=evoked.info['ch_names'],
        bads=[],
        projs=[],
        nfree=baseline_data.shape[1],
    )

    inverse_operator = make_inverse_operator(
        evoked.info,
        fwd,
        cov,
        loose=loose,
        depth=depth,
    )

    lambda2 = 1.0 / snr ** 2
    stc = apply_inverse(
        evoked,
        inverse_operator,
        lambda2=lambda2,
        method=method,
        pick_ori=None,
    )

    stc.subject = 'fsaverage'
    stc.subjects_dir = subjects_dir
    return stc


# %% ---- 2026-05-15 ------------------------
# Play ground
mode = 'MEG'  # 'MEG' or 'EEG'

cond_1_path = Path(f'./output/step-1-subjects-average/{mode}/1-epo-ave.fif')
cond_2_path = Path(f'./output/step-1-subjects-average/{mode}/2-epo-ave.fif')
cond_3_path = Path(f'./output/step-1-subjects-average/{mode}/3-epo-ave.fif')

cond_1p_path = Path(
    f'./output/step-1-subjects-average-proj/{mode}/1-withproj-epo-ave.fif')

evoked_1 = mne.read_evokeds(cond_1_path)[0]
evoked_2 = mne.read_evokeds(cond_2_path)[0]
evoked_3 = mne.read_evokeds(cond_3_path)[0]
evoked_1p = mne.read_evokeds(cond_1p_path)[0]

print(evoked_1, evoked_2, evoked_3, evoked_1p)

# Source estimation
stc = source_estimation_evoked(evoked_1)
brain = stc.plot(hemi='both')
input('Press enter to terminate.')


# %% ---- 2026-05-15 ------------------------
# Pending


# %% ---- 2026-05-15 ------------------------
# Pending
