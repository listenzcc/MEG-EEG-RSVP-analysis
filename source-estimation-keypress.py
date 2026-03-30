"""
File: source-estimation-keypress.py
Author: Chuncheng Zhang
Date: 2026-03-27
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for keypressing.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-27 ------------------------
# Requirements and constants
from util.easy_imports import *

from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator
from mne.datasets import fetch_fsaverage

# %%
DATA_DIR1 = Path(f'output/step-1-subjects-average')
assert DATA_DIR1.exists(), f'{DATA_DIR1} does not exist.'

# %% ---- 2026-03-27 ------------------------
# Function and class


# %% ---- 2026-03-27 ------------------------
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
evt = '3'
mode = 'EEG'
if len(sys.argv) > 1:
    mode = sys.argv[1].upper()
print(f'{mode=}')

evoked = mne.read_evokeds(
    DATA_DIR1 / mode / f'{evt}-epo-ave.fif')[0]

# evoked.filter(l_freq=0.01, h_freq=8)

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
# baseline_data = evoked.copy().crop(evoked.tmin, 0).data

if mode == 'MEG':
    baseline_data = evoked.copy().crop(1, 1.4).data
elif mode == 'EEG':
    baseline_data = evoked.copy().crop(evoked.tmin, -0.25).data

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

# 9. 可视化结果

# 9.1 绘制源时间序列
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
# 选择峰值时间点
peak_time = evoked.get_peak()[1]
stc.plot(hemi='both', subjects_dir=subjects_dir, subject='fsaverage')

input('Press enter to escape.')
exit(0)

# %% ---- 2026-03-27 ------------------------
# Pending


# %% ---- 2026-03-27 ------------------------
# Pending
