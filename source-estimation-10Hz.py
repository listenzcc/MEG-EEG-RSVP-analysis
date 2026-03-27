"""
File: source-estimation-10Hz.py
Author: Chuncheng Zhang
Date: 2026-03-25
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Source estimation for the 10Hz SSVEP component.

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2026-03-25 ------------------------
# Requirements and constants
from mne.minimum_norm import make_inverse_operator, apply_inverse, write_inverse_operator
from mne.datasets import fetch_fsaverage
from util.easy_imports import *

# %%
DATA_DIR = Path(f'output/step-1-subjects-average')
assert DATA_DIR.exists(), f'{DATA_DIR} does not exist.'

# %% ---- 2026-03-25 ------------------------
# Function and class


def plot_ssvep_source_power(evoked, fwd, noise_cov, freq=10.0, method='fft'):
    """
    计算 SSVEP evoked 在源空间的频率功率，并绘制脑表热图。

    Parameters
    ----------
    evoked : mne.Evoked
        平均后的 SSVEP 响应。
    fwd : mne.Forward
        前向解。
    noise_cov : mne.Covariance
        噪声协方差矩阵。
    freq : float
        要提取的频率（Hz），默认为 10。
    method : str
        提取功率的方法，可选 'hilbert' 或 'fft'。

    Returns
    -------
    stc_power : mne.SourceEstimate
        包含指定频率功率的源空间对象。
    """
    # 1. 构建逆解
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=0.2, depth=0.8
    )

    # 2. 源空间时间序列
    stc = mne.minimum_norm.apply_inverse(
        evoked, inverse_operator, lambda2=1/9., method='dSPM')

    # 3. 提取频率功率
    if method.lower() == 'fft':
        n_times = stc.data.shape[1]
        sfreq = evoked.info['sfreq']
        fft_data = np.fft.rfft(stc.data, axis=1)
        freqs = np.fft.rfftfreq(n_times, 1/sfreq)
        idx = np.argmin(np.abs(freqs - freq))
        power = np.abs(fft_data[:, idx])**2
    elif method.lower() == 'hilbert':
        stc_band = stc.copy().filter(freq-2, freq+2, fir_design='firwin')
        analytic = mne.filter.hilbert(stc_band.data, envelope=True)
        power = np.mean(analytic**2, axis=1)
    else:
        raise ValueError("method must be 'fft' or 'hilbert'")

    # 4. 构建单时间点 STC
    stc_power = stc.copy()
    stc_power.data = power[:, np.newaxis]

    return stc_power

# %% ---- 2026-03-25 ------------------------
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
evt = '2'
mode = 'MEG'
if len(sys.argv) > 1:
    m = sys.argv[1].upper()
    if m in ['MEG', 'EEG']:
        mode = sys.argv[1].upper()
print(f'{mode=}')

evoked = mne.read_evokeds(
    DATA_DIR / mode / f'{evt}-epo-ave.fif')[0]

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
# 使用0秒之前的数据计算协方差矩阵
baseline_data = evoked.copy().crop(evoked.tmin, 0).data
cov = mne.Covariance(
    data=np.cov(baseline_data),
    names=evoked.info['ch_names'],
    bads=[],
    projs=[],
    nfree=baseline_data.shape[1]
)

# %%
stc_power = plot_ssvep_source_power(evoked, fwd, cov)

brain = stc_power.plot(
    hemi='both', subjects_dir=subjects_dir, subject='fsaverage')
input('')

# %% ---- 2026-03-25 ------------------------
# Pending


# %% ---- 2026-03-25 ------------------------
# Pending
