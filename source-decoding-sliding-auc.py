"""
File: source-decoding-sliding-auc.py
Author: Chuncheng Zhang
Date: 2026-04-03
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Decoding sliding in source level.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-04-03 ------------------------
# Requirements and constants
import pickle
import json
from mne.datasets import fetch_fsaverage
from mne.decoding import SlidingEstimator, cross_val_multiscore, Vectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from util.easy_imports import *

# %%
SUBJ = 'S02'
MODE = 'EEG'

if len(sys.argv) > 1:
    _, SUBJ, MODE = sys.argv

logger.info(f'Run {__file__} for {SUBJ=}, {MODE=}')

# %%
DATA_DIR = Path(f'output/decoding-step-1/{MODE}-{SUBJ}')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist'

# %%
OUTPUT_DIR = Path(f'output/source-decoding-sliding-auc/{MODE}-{SUBJ}')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# %% ---- 2026-04-03 ------------------------
# Function and class


# %% ---- 2026-04-03 ------------------------
# Play ground
# target
epochs1 = mne.read_epochs(DATA_DIR / 'epochs-1-epo.fif', preload=True)
# non-target
epochs2 = mne.read_epochs(DATA_DIR / 'epochs-2-epo.fif', preload=True)

# 合并所有epochs数据
epochs = mne.concatenate_epochs([
    epochs1,  # [:30],
    epochs2,  # [:30],
])
# epochs = epochs[:200]
print(epochs)


# %%
if MODE == 'EEG':
    epochs.set_eeg_reference('average', projection=True)
    assert not epochs.info['custom_ref_applied'], "仍然有自定义参考"

# %%
# ==== SOURCE RECONSTRUCTION ====

# 1. 准备模板数据（使用fsaverage模板）
print("下载fsaverage模板...")
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# 2. 加载模板的MRI数据
trans = 'fsaverage'  # 使用模板的trans
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# 4. 创建正向算子（需要设置电极位置）
# 计算电极位置与模板的配准
# 使用标准电极位置配准到fsaverage
montage = epochs.get_montage()
if montage is None:
    print("evoked没有电极位置信息，使用标准1020系统")
    montage = mne.channels.make_standard_montage('standard_1020')
    epochs.set_montage(montage)

# 5. 计算正向解
print("计算正向解...")
# 创建源空间
src = mne.read_source_spaces(src)

# 计算导联场矩阵
fwd = mne.make_forward_solution(
    epochs.info,
    trans=trans,
    src=src,
    bem=bem,
    eeg=MODE == 'EEG',
    meg=MODE == 'MEG'
)
print(f"正向解计算完成: {fwd}")

noise_cov = mne.compute_covariance(epochs, tmax=0, method=['empirical'])

inverse_operator = mne.minimum_norm.make_inverse_operator(
    epochs.info, fwd, noise_cov, loose=0.2, depth=0.8
)

stcs = mne.minimum_norm.apply_inverse_epochs(
    epochs,
    inverse_operator,
    lambda2=1.0 / 9.0,
    method='dSPM',
    pick_ori="normal"
)

# print(stcs)

# %%
# ==== LABEL EXTRACTION ====

labels = mne.read_labels_from_annot(
    subject='fsaverage',
    parc='aparc.a2009s',
    subjects_dir=subjects_dir
)
labels = labels[:-2]

# ! Use less labels for testing
# labels = labels[:3]

y = epochs.events[:, -1]

scores_all_regions = {}

for label in tqdm(labels):
    mat = np.array([stc.in_label(label).data for stc in tqdm(stcs)])

    print(label, mat.shape)  # (n_epochs, n_vertices_in_label, n_times)
    clf = make_pipeline(
        Vectorizer(),
        StandardScaler(),
        SVC(kernel='rbf')
    )

    # 10-fold CV
    cv = StratifiedKFold(n_splits=10, shuffle=True)

    time_decod = SlidingEstimator(
        clf,
        scoring='roc_auc',
        n_jobs=-1
    )

    scores = cross_val_multiscore(
        time_decod,
        mat,
        y,
        cv=cv,
        n_jobs=-1
    )

    scores_all_regions[label] = scores
    fname = OUTPUT_DIR / 'regions' / f'scores_{label.name}.json'
    fname.parent.mkdir(exist_ok=True, parents=True)
    json.dump(scores.tolist(), open(fname, 'w'))
    logger.info(f'Saved scores for {label.name} to {fname}')


print(scores_all_regions)
for label, scores in scores_all_regions.items():
    print(label, scores.shape)  # (n_splits, n_times)

# %%
fname = OUTPUT_DIR / 'scores_all_regions.json'
_obj = {label.name: scores.tolist()
        for label, scores in scores_all_regions.items()}
with open(fname, 'w') as f:
    json.dump(_obj, f)
logger.info(f'Saved scores for all regions to {fname}')

fname = OUTPUT_DIR / 'scores_all_regions.dumps'
with open(fname, 'wb') as f:
    pickle.dump(scores_all_regions, f)
logger.info(f'Saved scores for all regions to {fname}')

# %% ---- 2026-04-03 ------------------------
# Pending

# %% ---- 2026-04-03 ------------------------
# Pending
