"""
File: source-compare-with-without-artificial-removal.py
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

from scipy.signal import hilbert
from matplotlib.backends.backend_pdf import PdfPages
from util.easy_imports import *
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.datasets import fetch_fsaverage

# print(__IPYTHON__)

# %%
OUTPUT_DIR = Path('./output/comparison-ERPs-with-without-artificial-removal')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Ensure fsaverage is available and use the standard fsaverage BEM/source space.
fs_dir = fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)


# %%

# %% ---- 2026-05-15 ------------------------
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

print([evoked_1, evoked_2, evoked_3, evoked_1p])

# %%
l_freq, h_freq = None, 8
filter_args = (l_freq, h_freq)
figsize = (8, 4)
fs = 200  # Hz

# Define frequencies for Morlet wavelets (e.g., 2–40 Hz, 0.5 Hz steps)
freqs = np.arange(1, 31, 0.5)   # 2 to 40 Hz in 0.5 Hz steps
n_cycles = freqs / 2.0          # number of cycles increases with frequency


def better_fig(fig, name='name'):
    fig.set_size_inches(*figsize)
    fig.suptitle(name)
    return fig


balls = {
    'Target (Raw)':
    dict(
        evoked=evoked_1.copy(),
        filter_args=(None, 8),
        picks=['MZO01'] if mode == 'MEG' else ['Oz']
    ),
    'Target (Proj)':
    dict(
        evoked=evoked_1p.copy(),
        filter_args=(None, 8),
        picks=['MZO01'] if mode == 'MEG' else ['Oz']
    ),
    'Keypress':
    dict(
        evoked=evoked_3.copy(),
        filter_args=(None, 8),
        picks=['MLC42'] if mode == 'MEG' else ['C3']
    ),
    'NonTarget (10)':
    dict(
        evoked=evoked_2.copy(),
        filter_args=(8, 12),
        picks=['MZO01'] if mode == 'MEG' else ['Oz']
    ),
    'NonTarget (20)':
    dict(
        evoked=evoked_2.copy(),
        filter_args=(18, 22),
        picks=['MZO01'] if mode == 'MEG' else ['Oz']
    ),
}

# %%
analysis_on_what_condition = 'NonTarget (10)'
analysis_on_what_condition = 'Keypress'
analysis_on_what_condition = 'Target (Proj)'
# analysis_on_what_condition = 'Target (Raw)'

# %%
# NonTarget (10)
if analysis_on_what_condition == 'NonTarget (10)':
    name = 'NonTarget (10)'
    ball = balls[name]

    name = f'{mode} | {name}'
    evoked = ball['evoked']
    evoked.filter(*ball['filter_args'])

    hdata = hilbert(evoked.data, axis=-1)
    hdata = np.abs(hdata) ** 2
    print(hdata.shape)

    evo = evoked.copy()
    evo.data = hdata
    evoked = evo

    # Prevent Customized Reference in EEG since it is not allowed in inverse solution.
    if mode == 'EEG':
        evoked.set_eeg_reference('average', projection=True)
        assert not evoked.info['custom_ref_applied'], "仍然有自定义参考"

    stc = source_estimation_evoked(evoked, use_eye_cov=True)
    stc.plot(hemi='both', title=name)

# %%
if analysis_on_what_condition in ['Keypress', 'Target (Proj)', 'Target (Raw)']:
    name = analysis_on_what_condition
    ball = balls[name]

    name = f'{mode} | {name}'
    evoked = ball['evoked']
    evoked.filter(*ball['filter_args'])

    stc = source_estimation_evoked(evoked, use_eye_cov=True)
    stc.plot(hemi='both', title=name)


# %%
input('Press enter to terminate.')
exit(0)

# %%

with PdfPages(OUTPUT_DIR / f'{mode}.pdf') as pdf:

    import matplotlib
    matplotlib.use('pdf')  # before importing pyplot # noqa

    def append_fig(fig):
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    for name, ball in balls.items():
        name = f'{mode} | {name}'
        evoked = ball['evoked']
        # picks = ball['picks']
        evoked.filter(*ball['filter_args'])

        p, _ = evoked.get_peak(mode='abs')
        picks = [p]

        fig = evoked.plot_joint(times='peaks', show=False)
        fig.suptitle(name)
        append_fig(fig)

        fig = evoked.plot(gfp=True, spatial_colors=True,
                          hline=[0.0], show=False)
        fig.suptitle(name)
        append_fig(fig)

        fig = evoked.plot(picks=picks, show=False)
        fig.suptitle(f'{name} | {picks}')
        append_fig(fig)

        tfr = evoked.compute_tfr(
            method='morlet',
            freqs=freqs,
            n_cycles=n_cycles,
            decim=1,
            picks=picks
        )

        # tfr is an instance of AverageTFR
        # Plot power (dB scale, baseline corrected)
        tfr.plot(
            title=f'{name} | Morlet TFR | {picks}',
            mode='db',  # decibel conversion
            cmap='RdBu_r',
            show=False
        )
        fig = plt.gcf()
        fig.suptitle(f'{name} | Morlet TFR | {picks}')
        append_fig(fig)

plt.show()

# %% Source estimation

# stc = source_estimation_evoked(evoked_1)
# brain = stc.plot(hemi='both')
# input('Press enter to terminate.')


# %% ---- 2026-05-15 ------------------------
# Pending

# %% ---- 2026-05-15 ------------------------
# Pending
# %%

# %%
