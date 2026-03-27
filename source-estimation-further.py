"""
File: source-estimation-further.py
Author: Chuncheng Zhang
Date: 2026-03-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for evoked.
    Further analysis for P300 and diff.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-19 ------------------------
# Requirements and constants
from util.easy_imports import *

from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator
from mne.datasets import fetch_fsaverage

# %%
DATA_DIR1 = Path(f'output/step-1-subjects-average-proj')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist.'

# %% ---- 2026-03-19 ------------------------
# Function and class


# %% ---- 2026-03-19 ------------------------
# Play ground
# 1. 准备模板数据（使用fsaverage模板）
print("下载fsaverage模板...")
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = os.path.dirname(fs_dir)

# 2. 加载模板的MRI数据
trans = 'fsaverage'  # 使用模板的trans
src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# 3. 加载evoked数据
evt = '1'
mode = 'MEG'
if len(sys.argv) > 1:
    m = sys.argv[1].upper()
    if m in ['MEG', 'EEG']:
        mode = sys.argv[1].upper()
print(f'{mode=}')

evoked = mne.read_evokeds(
    DATA_DIR1 / mode / f'{evt}-withproj-epo-ave.fif')[0]

evoked.filter(l_freq=0.01, h_freq=8)

if mode == 'EEG':
    evoked.set_eeg_reference('average', projection=True)
    assert not evoked.info['custom_ref_applied'], "仍然有自定义参考"

print(evoked)

# 4. 创建正向算子（需要设置电极位置）
# 计算电极位置与模板的配准
# 使用标准电极位置配准到fsaverage
montage = evoked.get_montage()
if montage is None:
    print("evoked没有电极位置信息，使用标准1020系统")
    montage = mne.channels.make_standard_montage('standard_1020')
    evoked.set_montage(montage)

# 5. 计算正向解
print("计算正向解...")
# 创建源空间
src = mne.read_source_spaces(src)

# 计算导联场矩阵
fwd = mne.make_forward_solution(
    evoked.info,
    trans=trans,
    src=src,
    bem=bem,
    eeg=mode == 'EEG',
    meg=mode == 'MEG'
)
print(f"正向解计算完成: {fwd}")

# 6. 计算噪声协方差矩阵
# 如果没有噪声数据，可以创建单位协方差
# cov = mne.Covariance(
#     data=np.eye(len(evoked.info['ch_names'])),
#     names=evoked.info['ch_names'],
#     bads=[],
#     projs=[],
#     nfree=1e10
# )
# 使用0秒之前的数据计算协方差矩阵
baseline_data = evoked.copy().crop(evoked.tmin, 0).data
cov = mne.Covariance(
    data=np.cov(baseline_data),
    names=evoked.info['ch_names'],
    bads=[],
    projs=[],
    nfree=baseline_data.shape[1]
)

# 7. 创建逆算子
print("创建逆算子...")
inverse_operator = make_inverse_operator(
    evoked.info,
    fwd,
    cov,
    loose=0.2,  # 松耦合约束
    depth=0.8   # 深度加权
)

# 8. 应用逆算子进行溯源
print("进行溯源计算...")
snr = 3.0  # 信噪比
lambda2 = 1.0 / snr ** 2

stc = apply_inverse(
    evoked,
    inverse_operator,
    lambda2=lambda2,
    method='MNE',  # 可以使用 'dSPM', 'sLORETA', 'eLORETA'
    pick_ori=None
)

print(f"溯源结果: {stc}")

# %%
# 9. 可视化结果
# stc.plot(hemi='both', subjects_dir=subjects_dir, subject='fsaverage')
# input('Press enter to escape.')
# exit(0)

# %%


def window_aggregate(stc, tmin, tmax, mode='mean_abs'):
    times = stc.times
    mask = (times >= tmin) & (times <= tmax)

    data = stc.data[:, mask]

    if mode == 'mean_abs':
        out = np.mean(np.abs(data), axis=1)
    elif mode == 'mean':
        out = np.mean(data, axis=1)
    elif mode == 'max_abs':
        out = np.max(np.abs(data), axis=1)
    else:
        raise ValueError(mode)

    return out  # shape: (n_vertices,)


A1 = window_aggregate(stc, 0.2, 0.4)
A2 = window_aggregate(stc, 0.4, 0.6)
D = A2 - A1


# %%
labels = mne.read_labels_from_annot(
    subject='fsaverage',
    parc='aparc.a2009s',
    subjects_dir=subjects_dir
)

print(labels)
print(D.shape)
print(stc.vertices[0].size)

# %%


def make_stc_map(values, stc):
    return mne.SourceEstimate(
        values[:, None],
        vertices=stc.vertices,
        tmin=0,
        tstep=1
    )


def roi_reduce_safe(stc_map, labels):
    roi_dict = {}

    for label in labels:
        if label.name.startswith('Unknown'):
            continue

        sub_stc = stc_map.in_label(label)

        if sub_stc.data.size == 0:
            continue  # 这个label在当前src中没有点

        roi_dict[label.name] = sub_stc.data.mean()

    return roi_dict


stc_A1 = make_stc_map(A1, stc)
stc_A2 = make_stc_map(A2, stc)
stc_D = make_stc_map(D,  stc)

roi_A1 = roi_reduce_safe(stc_A1, labels)
roi_A2 = roi_reduce_safe(stc_A2, labels)
roi_D = roi_reduce_safe(stc_D,  labels)


print(roi_A1)
print(roi_A2)
print(roi_D)

# %%
brain = stc_A1.plot(hemi='lh', subjects_dir=subjects_dir,
                    subject='fsaverage', backend='matplotlib')

brain = stc_A2.plot(hemi='lh', subjects_dir=subjects_dir,
                    subject='fsaverage', backend='matplotlib')

brain = stc_D.plot(hemi='lh', subjects_dir=subjects_dir,
                   subject='fsaverage', backend='matplotlib')

# %%

# %%

# %% ---- 2026-03-19 ------------------------
# Pending


# %% ---- 2026-03-19 ------------------------
# Pending
