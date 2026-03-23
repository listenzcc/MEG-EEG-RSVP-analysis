"""
File: source-estimation.py
Author: Chuncheng Zhang
Date: 2026-03-19
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for evoked.

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
evoked = mne.read_evokeds(
    DATA_DIR1 / mode / f'{evt}-withproj-epo-ave.fif')[0]

evoked.filter(l_freq=0.01, h_freq=8)
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
cov = mne.Covariance(
    data=np.eye(len(evoked.info['ch_names'])),
    names=evoked.info['ch_names'],
    bads=[],
    projs=[],
    nfree=1e10
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

# 9.2 绘制特定时间点的源分布
brain = stc.plot(
    subjects_dir=subjects_dir,
    subject='fsaverage',
    hemi='both',
    time_viewer=True,
    views=['lat', 'med'],
    initial_time=peak_time,
    size=(800, 400)
)

# 9.3 绘制源估计的统计图
# 选择感兴趣的时间窗口
time_window = (0.1, 0.3)  # 示例时间窗
stc_masked = stc.copy().crop(time_window[0], time_window[1])
stc_masked.plot(hemi='both', subjects_dir=subjects_dir, subject='fsaverage')

# 9.4 绘制不同脑区的源活动
labels = mne.read_labels_from_annot('fsaverage', subjects_dir=subjects_dir)

# 选择一些感兴趣的脑区
interest_labels = ['fusiform', 'precentral', 'postcentral']
for label_name in interest_labels:
    matching_labels = [l for l in labels if label_name in l.name]
    if matching_labels:
        label = matching_labels[0]
        label_stc = stc.in_label(label)

        plt.figure()
        plt.title(f'{label.name} 源活动')
        plt.plot(stc.times, label_stc.data.mean(axis=0))
        plt.xlabel('Time (s)')
        plt.ylabel('Source amplitude')
        plt.grid(True)

# 10. 保存结果
# 保存源估计
# stc.save('evoked_source_estimate')
# write_inverse_operator('inverse_operator.fif', inverse_operator)

# 11. 显示所有脑区的平均源活动
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

# 计算不同脑叶的平均活动
hemispheres = ['lh', 'rh']
lobes = ['frontal', 'parietal', 'temporal', 'occipital']

for idx, lobe in enumerate(lobes):
    for hemi in hemispheres:
        # 找到对应脑叶的标签
        matching_labels = [
            l for l in labels if lobe in l.name.lower() and hemi in l.name]
        if matching_labels:
            # 合并标签
            combined_label = mne.Label(
                vertices=np.unique(np.concatenate(
                    [l.vertices for l in matching_labels])),
                hemi=hemi,
                name=f'{hemi}_{lobe}'
            )
            # 提取该区域的源活动
            region_stc = stc.in_label(combined_label)
            axes[idx].plot(stc.times, region_stc.data.mean(axis=0),
                           label=hemi, linewidth=2)

    axes[idx].set_title(f'{lobe.capitalize()} Lobe')
    axes[idx].set_xlabel('Time (s)')
    axes[idx].set_ylabel('Source amplitude')
    axes[idx].legend()
    axes[idx].grid(True)

plt.tight_layout()
plt.show()

print("溯源分析完成！")
# %% ---- 2026-03-19 ------------------------
# Pending


# %% ---- 2026-03-19 ------------------------
# Pending
