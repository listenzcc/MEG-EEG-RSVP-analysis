"""
File: source-estimation-on-evoked.py
Author: Chuncheng Zhang
Date: 2026-06-02
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for given evoked file.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-06-02 ------------------------
# Requirements and constants
import argparse
from scipy.stats import norm, median_abs_deviation
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from statsmodels.stats.multitest import fdrcorrection

from util.easy_imports import *

# %%
parser = argparse.ArgumentParser(
    description='Source estimation for given evoked file.')
parser.add_argument('--mode', type=str,
                    choices=['EEG', 'MEG'], required=True, help='Data modality: EEG or MEG')
parser.add_argument('--file', type=str, required=True,
                    help='Evoked file path (e.g., output/EEG-evoked-quick-by-diff-times-ave.fif)')
parser.add_argument('--output', type=str, default=None,
                    help='Output file path for the source estimate.')

args = parser.parse_args()

options = args.__dict__
print(f'{options=}')

# %%
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)

# %% ---- 2026-06-02 ------------------------
# Function and class


def source_estimation_evoked(evoked, method='MNE', snr=3.0, loose=0.2, depth=0.8, baseline=(None, 0.0), use_eye_cov=False):
    """Compute a source estimate for an MNE Evoked using fsaverage BEM."""
    evoked = evoked.copy()

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

    # Use eye covariance if no baseline data is available (not recommended).
    # Used for SSVEP in all times.
    baseline_evoked = evoked.copy().crop(tmin, tmax)
    baseline_data = baseline_evoked.data
    if use_eye_cov:
        cov = mne.Covariance(
            data=np.eye(len(evoked.info['ch_names'])),  # --- IGNORE ---
            names=evoked.info['ch_names'],
            bads=[],
            projs=[],
            nfree=baseline_data.shape[1],
        )
    else:
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

    # Remove all the values from the unknown areas in aparc_sub atlas.
    # The offset is needed since the labels are concatenated from both hemispheres.
    # The offset value is the number of vertices in the left hemisphere,
    # which is the same as the number of vertices in the right hemisphere in fsaverage.
    parc = 'aparc_sub'
    labels_parc = mne.read_labels_from_annot(
        'fsaverage', parc=parc, subjects_dir=subjects_dir)
    labels_parc_df = pd.DataFrame([(e.name, e)
                                   for e in labels_parc], columns=['name', 'label'])
    labels_parc_df.set_index('name', inplace=True)
    stc1 = stc.in_label(labels_parc_df.loc['unknown-lh', 'label'])
    stc2 = stc.in_label(labels_parc_df.loc['unknown-rh', 'label'])
    offset = len(stc.vertices[0])
    for v in np.concat((stc1.vertices[0], stc2.vertices[1] + offset)):
        stc.data[v, :] = 0

    stc.subject = 'fsaverage'
    stc.subjects_dir = subjects_dir
    return stc


def z_to_p(z_scores):
    return 2 * (1 - norm.cdf(np.abs(z_scores)))


def fdr_corrected_mask(p_values, q=0.05):
    p_flat = p_values.ravel()
    reject, _ = fdrcorrection(p_flat, alpha=q)
    return reject.reshape(p_values.shape)


def convert_into_zscore(stc):
    '''
    Convert the source estimate data into z-scores using median and MAD.
    This is a robust method to identify significant activations while mitigating the influence of outliers.

    Args:
        stc (mne.SourceEstimate): An MNE SourceEstimate object containing the source data to be converted.

    Returns:
        mne.SourceEstimate: The input SourceEstimate object with its data converted into z-scores
    '''

    print(
        f'Converting source estimate {stc} into z-scores using median and MAD...')
    data = stc.data
    values = data.ravel()
    # 稳健估计背景分布
    median_val = np.median(values)
    # 或使用 np.median(np.abs(values - median_val))
    mad_val = median_abs_deviation(values)

    # 避免 MAD 为零（极端情况）
    if mad_val == 0:
        mad_val = 1e-8

    # 计算每个时空点的 z 分数
    z_scores = (data - median_val) / mad_val

    stc.data = z_scores
    return stc


def statistic_to_stc(stc):
    '''
    Convert the source estimate data into z-scores and apply FDR correction to identify significant activations.
    This function combines the robust z-score conversion with FDR correction to control for multiple comparisons.

    Args:
        stc (mne.SourceEstimate): An MNE SourceEstimate object containing the source data

    Returns:
        mne.SourceEstimate: The input SourceEstimate object with its data converted into z-scores and non-significant activations set to zero
    '''

    stc = convert_into_zscore(stc)
    p_values = z_to_p(stc.data)
    mask = fdr_corrected_mask(p_values, q=0.05)
    stc.data[~mask] = 0
    return stc


# %% ---- 2026-06-02 ------------------------
# Play ground
evoked = mne.read_evokeds(options['file'])[0]

# Prevent Customized Reference in EEG since it is not allowed in inverse solution.
if options['mode'] == 'EEG':
    evoked.set_eeg_reference('average', projection=True)
    assert not evoked.info['custom_ref_applied'], "仍然有自定义参考"

print(evoked.times[0], evoked.times[-1])

stc = source_estimation_evoked(evoked, method='MNE', snr=3.0,
                               loose=0.2, depth=0.8, baseline=(None, -0.35), use_eye_cov=False)
print(stc)
# statistic_to_stc(stc)
brain = stc.plot(title=f'Source Estimation for {options["file"]}')

if options['output'] is not None:
    stc.save(options['output'], overwrite=True)
    print(f'Saved source estimate to {options["output"]}')

print(options['file'])

brain.show()

plt.plot([1, 2, 3], [4, 5, 6])  # Dummy plot to keep the window open
plt.show(block=True)
# input("Press Enter to quit ...")

# %% ---- 2026-06-02 ------------------------
# Pending


# %% ---- 2026-06-02 ------------------------
# Pending
